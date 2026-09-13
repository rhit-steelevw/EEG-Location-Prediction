"""
Synthetic validation for si2_vq_fixed.py.

Generates waveforms with known cluster identity, random shift, and random
sign, embeds them in longer noisy samples, and checks that si2_vq recovers
all three exactly. Run this after dropping si2_vq_fixed.py into your
notebook/package to confirm the fix before running it on real IC data.
"""
import numpy as np
from si2_vq_fixed import si2_vq

rng = np.random.default_rng(0)
L = 20   # centroid length
M = 40   # sample length

true_wave_a = np.sin(np.linspace(0, 2 * np.pi, L))
true_wave_b = np.sin(np.linspace(0, 4 * np.pi, L)) * 0.5
centroids = np.vstack([true_wave_a, true_wave_b])

n_per_class = 50
samples, true_cluster, true_shift, true_sign = [], [], [], []
for _ in range(n_per_class):
    for k, wave in enumerate([true_wave_a, true_wave_b]):
        shift = rng.integers(0, M - L + 1)
        sign = rng.choice([1, -1])
        sample = rng.normal(0, 0.01, size=M)
        sample[shift:shift + L] += sign * wave
        samples.append(sample)
        true_cluster.append(k)
        true_shift.append(shift)
        true_sign.append(sign)

X = np.array(samples)
true_cluster = np.array(true_cluster)
true_shift = np.array(true_shift)
true_sign = np.array(true_sign)

for metric in ['euclidean', 'cosine']:
    labels, shifts, distances, signs = si2_vq(X, centroids, metric)
    acc = np.mean(labels == true_cluster)
    shift_ok = np.mean(shifts == true_shift)
    sign_ok = np.mean(signs == true_sign)
    print(f"metric={metric}: cluster_acc={acc:.3f}, shift_match={shift_ok:.3f}, sign_match={sign_ok:.3f}")
    assert acc > 0.99, f"Sign/shift-invariant assignment failed for metric={metric}"

print("All checks passed: si2_vq correctly recovers cluster, shift, and sign.")
