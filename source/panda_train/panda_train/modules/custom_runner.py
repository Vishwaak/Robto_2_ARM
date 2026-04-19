import torch
import torch.nn as nn
import wandb
from rsl_rl.algorithms.distillation import Distillation


class DistillationWithAux(Distillation):
    """Distillation + auxiliary object position prediction loss."""

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_behavior_loss = 0.0
        mean_aux_loss = 0.0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()

            for batch in self.storage.generator():
                actions = self.student(batch.observations)
                behavior_loss = self.loss_fn(actions, batch.privileged_actions)

                # Aux loss — object_pos is last 3 dims of teacher obs
                aux_loss = torch.tensor(0.0, device=self.device)
                if (
                    hasattr(self.student, "_aux_pred")
                    and self.student._aux_pred is not None
                ):
                    object_pos_gt = batch.observations["teacher"][:, -3:]
                    aux_loss = nn.functional.huber_loss(
                        self.student._aux_pred, object_pos_gt
                    )

                    if self.num_updates % 10 == 0 and cnt == 0 and epoch == 0:
                        pred = self.student._aux_pred[0].detach()
                        gt = object_pos_gt[0].detach()
                        print(f"[AUX] pred={pred.cpu().numpy().round(3)}  gt={gt.cpu().numpy().round(3)}  err={((pred-gt).abs().mean()).item():.4f}")
                

                loss = loss + behavior_loss + self.student.aux_weight * aux_loss
                mean_behavior_loss += behavior_loss.item()
                mean_aux_loss += aux_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(
                            self.student.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.student.detach_hidden_state()
                    loss = 0

                self.student.reset(batch.dones.view(-1))
                self.teacher.reset(batch.dones.view(-1))
                self.student.detach_hidden_state(batch.dones.view(-1))

        mean_behavior_loss /= cnt
        mean_aux_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = (
            self.student.get_hidden_state(),
            self.teacher.get_hidden_state(),
        )
        self.student.detach_hidden_state()

        if wandb.run is not None:
            wandb.log({
                "distillation/behavior_loss": mean_behavior_loss,
                "distillation/aux_loss": mean_aux_loss,
            }, step=self.num_updates, commit=False)
        
        if self.num_updates % 10 == 0 and self.student._aux_pred is not None:
            pred = self.student._aux_pred[0].detach()
            gt = object_pos_gt[0].detach()
            print(f"[AUX] pred={pred.cpu().numpy().round(3)}  gt={gt.cpu().numpy().round(3)}  err={((pred-gt).abs().mean()).item():.4f}")

        return {"behavior": mean_behavior_loss, "aux": mean_aux_loss}



from rsl_rl.algorithms.ppo import PPO


class PPOWithAux(PPO):
    """PPO with auxiliary object position prediction loss from CNN latent."""

    def __init__(self, *args, aux_coef: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_coef = aux_coef
        self.num_updates = 0

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_aux_loss = 0

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            # Actor forward — populates _aux_pred via CNNModelWithAux.get_latent
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Adaptive LR
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # Aux loss — object_pos is last 3 dims of teacher obs
            aux_loss = torch.tensor(0.0, device=self.device)
            if (
                hasattr(self.actor, "_aux_pred")
                and self.actor._aux_pred is not None
                and "teacher" in batch.observations.keys()
            ):
                object_pos_gt = batch.observations["teacher"][:, -3:]
                aux_loss = nn.functional.huber_loss(self.actor._aux_pred, object_pos_gt)
                loss = loss + self.aux_coef * aux_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_aux_loss += aux_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_aux_loss /= num_updates
       
        self.storage.clear()

        if wandb.run is not None:
            wandb.log({
                "train/aux_loss": mean_aux_loss,
                "train/value_loss": mean_value_loss,
                "train/surrogate_loss": mean_surrogate_loss,
            }, step=self.num_updates, commit=False)
        self.num_updates += 1
        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "aux": mean_aux_loss,
        }