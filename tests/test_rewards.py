import torch

from rlformer.rewards import (coherence_factor_reward, feature_sharpness_reward,
                              local_texture_entropy_reward, unsupervised_reward)


def test_coherence_factor_bounds():
    aligned = torch.ones(50, 16)
    noise = torch.randn(50, 16)
    assert torch.isclose(coherence_factor_reward(aligned), torch.tensor(1.0), atol=1e-4)
    assert 0.0 <= coherence_factor_reward(noise) < 0.5


def test_sharpness_prefers_sharper_images():
    sharp = torch.zeros(40, 40)
    sharp[:, 20:] = 1.0
    blurred = torch.nn.functional.avg_pool2d(sharp[None, None], 9, stride=1, padding=4)[0, 0]
    assert feature_sharpness_reward(sharp) > feature_sharpness_reward(blurred)


def test_texture_entropy_prefers_structure_over_noise():
    torch.manual_seed(0)
    structured = torch.zeros(30, 30)
    structured[:, 15:] = 1.0
    noise = torch.rand(30, 30)
    assert local_texture_entropy_reward(structured) > local_texture_entropy_reward(noise)


def test_combined_reward_is_differentiable():
    img = torch.randn(30, 30, requires_grad=True)
    ch = torch.randn(900, 8, requires_grad=True)
    reward, parts = unsupervised_reward(img, ch)
    reward.backward()
    assert torch.isfinite(reward) and set(parts) == {"fsr", "lte", "cf"}
    assert torch.isfinite(img.grad).all() and torch.isfinite(ch.grad).all()
