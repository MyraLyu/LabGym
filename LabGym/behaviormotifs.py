'''
Window-based behavioral motif discovery for LabGym.

Each observation is a sliding-window behavior composition vector rather than a
single frame label. A diagonal-covariance Gaussian HMM is fitted to the ordered
window vectors, then decoded window states are mapped back to every frame.
'''

from __future__ import annotations

from collections import Counter
import json
import os

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

_EPS = 1e-12


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
	matrix = np.asarray(matrix, dtype=float)
	denominator = matrix.sum(axis=1, keepdims=True)
	denominator[denominator <= 0] = 1.0
	return matrix / denominator


def _fill_missing_behaviors(events: Sequence[Sequence[object]]) -> Tuple[List[str], int]:
	labels = [str(item[0]) if len(item) > 0 else 'NA' for item in events]
	valid = [i for i, label in enumerate(labels) if label != 'NA']
	if not valid:
		raise ValueError('The selected animal has no valid behavior labels.')
	missing_count = sum(label == 'NA' for label in labels)
	first = valid[0]
	for i in range(first):
		labels[i] = labels[first]
	for i in range(first + 1, len(labels)):
		if labels[i] == 'NA':
			labels[i] = labels[i - 1]
	return labels, missing_count


def _encode(labels: Sequence[str], behavior_names: Sequence[str]) -> np.ndarray:
	mapping = {name: i for i, name in enumerate(behavior_names)}
	try:
		return np.asarray([mapping[label] for label in labels], dtype=np.int64)
	except KeyError as exc:
		raise ValueError('Behavior label is absent from behavior_names: ' + str(exc)) from exc


def _window_compositions(
	observations: np.ndarray,
	n_behaviors: int,
	window_size: int,
	window_step: int,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
	'''Convert frame labels into ordered sliding-window behavior histograms.'''
	total_frames = len(observations)
	window_size = max(2, min(int(window_size), total_frames))
	window_step = max(1, int(window_step))

	starts = list(range(0, max(total_frames - window_size + 1, 1), window_step))
	last_start = max(0, total_frames - window_size)
	if not starts or starts[-1] != last_start:
		starts.append(last_start)

	features = []
	centers = []
	bounds = []
	for start in starts:
		end = min(start + window_size, total_frames)
		counts = np.bincount(observations[start:end], minlength=n_behaviors).astype(float)
		counts /= max(counts.sum(), 1.0)
		features.append(counts)
		centers.append((start + end - 1) // 2)
		bounds.append((start, end - 1))

	return np.asarray(features, dtype=float), np.asarray(centers, dtype=int), bounds


class GaussianHMM:
	'''Diagonal-covariance Gaussian HMM fitted with Baum-Welch EM.'''

	def __init__(
		self,
		n_components: int,
		n_features: int,
		random_seed: int = 42,
		max_iter: int = 200,
		tol: float = 1e-4,
		pseudocount: float = 1e-2,
		min_covar: float = 1e-4,
	):
		if n_components < 2:
			raise ValueError('n_components must be at least 2.')
		self.n_components = int(n_components)
		self.n_features = int(n_features)
		self.random_seed = int(random_seed)
		self.max_iter = int(max_iter)
		self.tol = float(tol)
		self.pseudocount = float(pseudocount)
		self.min_covar = float(min_covar)
		self.startprob_ = None
		self.transmat_ = None
		self.means_ = None
		self.covars_ = None
		self.log_likelihood_history_ = []

	def _initialize(self, x: np.ndarray) -> None:
		rng = np.random.default_rng(self.random_seed)
		m = self.n_components
		t = len(x)
		self.startprob_ = np.full(m, 1.0 / m)
		transition = rng.gamma(1.0, 1.0, size=(m, m)) + np.eye(m) * 5.0
		self.transmat_ = _normalize_rows(transition)

		if t >= m:
			indices = np.linspace(0, t - 1, m, dtype=int)
			self.means_ = x[indices].copy()
		else:
			self.means_ = x[rng.integers(0, t, size=m)].copy()
		self.means_ += rng.normal(0.0, 0.01, size=self.means_.shape)
		global_var = np.var(x, axis=0) + self.min_covar
		self.covars_ = np.tile(global_var, (m, 1))

	def _log_emission(self, x: np.ndarray) -> np.ndarray:
		variance = np.clip(self.covars_, self.min_covar, None)
		difference = x[:, None, :] - self.means_[None, :, :]
		return -0.5 * (
			self.n_features * np.log(2.0 * np.pi)
			+ np.sum(np.log(variance), axis=1)[None, :]
			+ np.sum((difference ** 2) / variance[None, :, :], axis=2)
		)

	def _forward_backward(self, x: np.ndarray):
		log_start = np.log(np.clip(self.startprob_, _EPS, 1.0))
		log_trans = np.log(np.clip(self.transmat_, _EPS, 1.0))
		log_emit = self._log_emission(x)
		t, m = log_emit.shape
		alpha = np.empty((t, m), dtype=float)
		alpha[0] = log_start + log_emit[0]
		for i in range(1, t):
			alpha[i] = log_emit[i] + logsumexp(alpha[i - 1][:, None] + log_trans, axis=0)
		likelihood = float(logsumexp(alpha[-1]))
		beta = np.zeros((t, m), dtype=float)
		for i in range(t - 2, -1, -1):
			beta[i] = logsumexp(log_trans + log_emit[i + 1][None, :] + beta[i + 1][None, :], axis=1)
		gamma = np.exp(alpha + beta - likelihood)
		gamma /= np.clip(gamma.sum(axis=1, keepdims=True), _EPS, None)
		return alpha, beta, gamma, likelihood, log_trans, log_emit

	def fit(self, x: np.ndarray) -> 'GaussianHMM':
		x = np.asarray(x, dtype=float)
		if x.ndim != 2 or len(x) < 2:
			raise ValueError('At least two window observations are required.')
		self._initialize(x)
		previous = -np.inf
		for _ in range(self.max_iter):
			alpha, beta, gamma, likelihood, log_trans, log_emit = self._forward_backward(x)
			self.log_likelihood_history_.append(likelihood)
			xi_sum = np.zeros((self.n_components, self.n_components), dtype=float)
			for i in range(len(x) - 1):
				log_xi = alpha[i][:, None] + log_trans + log_emit[i + 1][None, :] + beta[i + 1][None, :] - likelihood
				xi_sum += np.exp(log_xi)
			self.startprob_ = gamma[0] + self.pseudocount
			self.startprob_ /= self.startprob_.sum()
			self.transmat_ = _normalize_rows(xi_sum + self.pseudocount)
			weights = np.clip(gamma.sum(axis=0), _EPS, None)
			self.means_ = (gamma.T @ x) / weights[:, None]
			diff = x[:, None, :] - self.means_[None, :, :]
			self.covars_ = np.sum(gamma[:, :, None] * diff ** 2, axis=0) / weights[:, None]
			self.covars_ = np.clip(self.covars_, self.min_covar, None)
			if np.isfinite(previous) and abs(likelihood - previous) < self.tol:
				break
			previous = likelihood
		return self

	def predict(self, x: np.ndarray) -> np.ndarray:
		x = np.asarray(x, dtype=float)
		log_start = np.log(np.clip(self.startprob_, _EPS, 1.0))
		log_trans = np.log(np.clip(self.transmat_, _EPS, 1.0))
		log_emit = self._log_emission(x)
		t, m = log_emit.shape
		delta = np.empty((t, m), dtype=float)
		psi = np.zeros((t, m), dtype=np.int64)
		delta[0] = log_start + log_emit[0]
		for i in range(1, t):
			candidate = delta[i - 1][:, None] + log_trans
			psi[i] = np.argmax(candidate, axis=0)
			delta[i] = candidate[psi[i], np.arange(m)] + log_emit[i]
		states = np.empty(t, dtype=np.int64)
		states[-1] = int(np.argmax(delta[-1]))
		for i in range(t - 2, -1, -1):
			states[i] = psi[i + 1, states[i + 1]]
		return states


def _map_window_states_to_frames(window_states: np.ndarray, centers: np.ndarray, n_frames: int) -> np.ndarray:
	'''Assign every frame the state of its nearest decoded window center.'''
	frame_states = np.empty(n_frames, dtype=np.int64)
	boundaries = ((centers[:-1] + centers[1:]) / 2.0).astype(float)
	for frame in range(n_frames):
		index = int(np.searchsorted(boundaries, frame, side='right'))
		frame_states[frame] = window_states[index]
	return frame_states


def _segments(states: np.ndarray) -> List[Tuple[int, int, int]]:
	segments = []
	start = 0
	for index in range(1, len(states) + 1):
		if index == len(states) or states[index] != states[start]:
			segments.append((int(states[start]), start, index - 1))
			start = index
	return segments


def _collapse_labels(labels: Iterable[str]) -> List[str]:
	result = []
	for label in labels:
		if not result or result[-1] != label:
			result.append(label)
	return result


def _motif_composition(states: np.ndarray, observations: np.ndarray, n_motifs: int, n_behaviors: int) -> np.ndarray:
	composition = np.zeros((n_motifs, n_behaviors), dtype=float)
	for motif in range(n_motifs):
		selected = observations[states == motif]
		if len(selected) > 0:
			composition[motif] = np.bincount(selected, minlength=n_behaviors)
		composition[motif] += 1e-6
	composition = _normalize_rows(composition)
	return composition


def _motif_statistics(states, labels, time_points, composition, transition, behavior_names, top_n):
	segments = _segments(states)
	total = len(states)
	frame_step = float(np.median(np.diff(np.asarray(time_points, dtype=float)))) if len(time_points) > 1 else 1.0
	rows = []
	for motif in range(composition.shape[0]):
		motif_segments = [(start, end) for state, start, end in segments if state == motif]
		durations = [(end - start + 1) * frame_step for start, end in motif_segments]
		top_indices = np.argsort(composition[motif])[::-1][:top_n]
		top_behaviors = '; '.join(f'{behavior_names[i]} ({composition[motif, i]:.3f})' for i in top_indices)
		sequence_counter = Counter()
		for start, end in motif_segments:
			collapsed = _collapse_labels(labels[start:end + 1])[:max(3, top_n)]
			if collapsed:
				sequence_counter[' -> '.join(collapsed)] += 1
		dominant_sequence = sequence_counter.most_common(1)[0][0] if sequence_counter else ''
		incoming = transition[:, motif].copy(); outgoing = transition[motif].copy()
		incoming[motif] = -1; outgoing[motif] = -1
		preceding = int(np.argmax(incoming)); following = int(np.argmax(outgoing))
		representative = sorted(motif_segments, key=lambda pair: pair[1] - pair[0], reverse=True)[:3]
		intervals = '; '.join(f'{time_points[start]:.3f}-{time_points[end]:.3f}s' for start, end in representative)
		rows.append({
			'motif_id': motif + 1,
			'motif_name': f'Motif {motif + 1}',
			'dominant_behavior_sequence': dominant_sequence,
			'representative_behaviors': top_behaviors,
			'occupancy': float(np.mean(states == motif)),
			'frequency_per_minute': len(motif_segments) / max(total * frame_step / 60.0, _EPS),
			'mean_duration_seconds': float(np.mean(durations)) if durations else 0.0,
			'number_of_occurrences': len(motif_segments),
			'common_preceding_motif': preceding + 1,
			'common_following_motif': following + 1,
			'representative_time_intervals': intervals,
		})
	return pd.DataFrame(rows)


def _plot_timeline(time_points, states, output_path, animal_id, motif_names=None):
	fig, ax = plt.subplots(figsize=(14, 3.2))
	ax.imshow(states[np.newaxis, :] + 1, aspect='auto', interpolation='nearest', extent=[time_points[0], time_points[-1], 0, 1])
	ax.set_yticks([]); ax.set_xlabel('Time (seconds)')
	ax.set_title(f'Behavioral Motif Timeline - ID {animal_id}')
	fig.tight_layout(); fig.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close(fig)


def _plot_composition(composition, behavior_names, output_path, animal_id, motif_names=None):
	width = max(8.0, 0.7 * len(behavior_names)); height = max(4.0, 0.65 * composition.shape[0])
	fig, ax = plt.subplots(figsize=(width, height))
	image = ax.imshow(composition, aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0, cmap='inferno')
	ax.set_xticks(np.arange(len(behavior_names))); ax.set_xticklabels(behavior_names, rotation=45, ha='right')
	ax.set_yticks(np.arange(composition.shape[0]))
	ax.set_yticklabels(motif_names if motif_names is not None else [f'Motif {i + 1}' for i in range(composition.shape[0])])
	ax.set_xlabel('Observed behavior'); ax.set_ylabel('Hidden motif')
	ax.set_title(f'Motif Composition Heatmap - ID {animal_id}')
	fig.colorbar(image, ax=ax, label='Behavior composition probability')
	fig.tight_layout(); fig.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close(fig)


def discover_behavioral_motifs(
	event_probability,
	time_points,
	behavior_names,
	output_folder,
	n_motifs,
	random_seed=42,
	max_iter=200,
	tol=1e-4,
	top_n_behaviors=3,
	window_size=30,
	window_step=5,
):
	"""
	Fit a sliding-window behavior-composition Gaussian HMM separately
	for every animal ID.

	Parameters
	----------
	event_probability : dict or list
		Behavior event sequences for one or more animal IDs.

		Dictionary format:
		{
			0: [...],
			1: [...],
		}

		List format:
		[
			[...],  # ID 0
			[...],  # ID 1
		]

	time_points : sequence
		Time point corresponding to each analyzed frame.

	behavior_names : sequence of str
		Ordered behavior class names.

	output_folder : str
		Folder in which motif outputs are saved.

	n_motifs : int
		Number of hidden behavioral motifs.

	random_seed : int
		Random seed used for reproducible model initialization.

	max_iter : int
		Maximum number of HMM fitting iterations.

	tol : float
		Convergence tolerance.

	top_n_behaviors : int
		Number of representative behaviors included in each motif summary.

	window_size : int
		Number of frames included in each sliding window.

	window_step : int
		Number of frames between consecutive sliding-window starts.

	Returns
	-------
	dict
		Results indexed by animal ID.
	"""

	if event_probability is None:
		raise ValueError(
			'No animal sequences were found in the input file.'
		)

	if isinstance(event_probability, dict):

		if len(event_probability) == 0:
			raise ValueError(
				'No animal sequences were found in the input file.'
			)

		event_sequences = list(event_probability.items())

	elif isinstance(event_probability, (list, tuple)):

		if len(event_probability) == 0:
			raise ValueError(
				'No animal sequences were found in the input file.'
			)

		# A single flat sequence may be returned as:
		# ['walking', 'walking', 'sniffing', ...]
		#
		# In that case, treat the whole list as animal ID 0.
		first_item = event_probability[0]

		if isinstance(
			first_item,
			(
				str,
				int,
				float,
				np.integer,
				np.floating,
			),
		) or first_item is None:

			event_sequences = [
				(0, event_probability)
			]

		else:

			event_sequences = list(
				enumerate(event_probability)
			)

	else:

		raise TypeError(
			'event_probability must be a dictionary, list, or tuple; '
			f'got {type(event_probability).__name__}.'
		)

	time_points = np.asarray(
		time_points,
		dtype=float,
	)

	if len(time_points) < 2:
		raise ValueError(
			'At least two time points are required.'
		)

	behavior_names = [
		str(name)
		for name in behavior_names
	]

	if len(behavior_names) < 2:
		raise ValueError(
			'At least two behavior classes are required.'
		)

	n_motifs = int(n_motifs)
	window_size = int(window_size)
	window_step = int(window_step)
	random_seed = int(random_seed)
	max_iter = int(max_iter)
	top_n_behaviors = int(top_n_behaviors)

	if n_motifs < 2:
		raise ValueError(
			'The number of motifs must be at least 2.'
		)

	if window_size < 2:
		raise ValueError(
			'Window size must be at least 2 frames.'
		)

	if window_step < 1:
		raise ValueError(
			'Window step must be at least 1 frame.'
		)

	if window_size > len(time_points):
		raise ValueError(
			'Window size cannot exceed the total number of frames. '
			f'Window size: {window_size}; '
			f'frames: {len(time_points)}.'
		)

	os.makedirs(
		output_folder,
		exist_ok=True,
	)

	all_results = {}

	for animal_id, events in event_sequences:

		events = list(events)

		if len(events) != len(time_points):

			raise ValueError(
				f'ID {animal_id}: event and time sequence lengths '
				f'do not match. Events: {len(events)}; '
				f'time points: {len(time_points)}.'
			)

		labels, missing_count = _fill_missing_behaviors(
			events
		)

		observations = _encode(
			labels,
			behavior_names,
		)

		features, centers, bounds = _window_compositions(
			observations,
			len(behavior_names),
			window_size,
			window_step,
		)

		if len(features) == 0:

			raise ValueError(
				f'ID {animal_id}: no sliding windows were generated. '
				'Reduce the window size or check the input sequence.'
			)

		if n_motifs > len(features):

			raise ValueError(
				f'ID {animal_id}: the number of motifs '
				f'({n_motifs}) exceeds the number of sliding '
				f'windows ({len(features)}).'
			)

		model = GaussianHMM(
			n_motifs,
			len(behavior_names),
			random_seed,
			max_iter,
			tol,
		).fit(features)

		window_states = model.predict(
			features
		)

		states = _map_window_states_to_frames(
			window_states,
			centers,
			len(observations),
		)

		composition = _motif_composition(
			states,
			observations,
			n_motifs,
			len(behavior_names),
		)

		prefix = (
			f'behavioral_motifs_ID{animal_id}'
		)

		# --------------------------------------------------------
		# Frame-level motif assignments
		# --------------------------------------------------------

		assignment = pd.DataFrame({
			'frame': np.arange(
				len(states),
				dtype=int,
			),
			'time_seconds': time_points,
			'observed_behavior': labels,
			'motif_id': states + 1,
			'motif_name': [
				f'Motif {state + 1}'
				for state in states
			],
		})

		assignment.to_csv(
			os.path.join(
				output_folder,
				prefix + '_frame_assignments.csv',
			),
			index=False,
		)

		# --------------------------------------------------------
		# Sliding-window composition and motif assignments
		# --------------------------------------------------------

		window_table = pd.DataFrame(
			features,
			columns=behavior_names,
		)

		window_table.insert(
			0,
			'window_end_frame',
			[
				end
				for _, end in bounds
			],
		)

		window_table.insert(
			0,
			'window_start_frame',
			[
				start
				for start, _ in bounds
			],
		)

		window_table.insert(
			0,
			'window_center_frame',
			centers,
		)

		window_table['motif_id'] = (
			window_states + 1
		)

		window_table['motif_name'] = [
			f'Motif {state + 1}'
			for state in window_states
		]

		window_table.to_csv(
			os.path.join(
				output_folder,
				prefix + '_window_assignments.csv',
			),
			index=False,
		)

		# --------------------------------------------------------
		# Motif summary
		# --------------------------------------------------------

		summary = _motif_statistics(
			states,
			labels,
			time_points,
			composition,
			model.transmat_,
			behavior_names,
			max(
				1,
				top_n_behaviors,
			),
		)

		summary.to_csv(
			os.path.join(
				output_folder,
				prefix + '_summary.csv',
			),
			index=False,
		)

		# --------------------------------------------------------
		# Motif composition probabilities
		# --------------------------------------------------------

		emission_table = pd.DataFrame(
			composition,
			index=[
				f'Motif {i + 1}'
				for i in range(n_motifs)
			],
			columns=behavior_names,
		)

		emission_table.to_csv(
			os.path.join(
				output_folder,
				prefix
				+ '_emission_probabilities.csv',
			)
		)

		# --------------------------------------------------------
		# Motif transition probabilities
		# --------------------------------------------------------

		motif_labels = [
			f'Motif {i + 1}'
			for i in range(n_motifs)
		]

		transition_table = pd.DataFrame(
			model.transmat_,
			index=motif_labels,
			columns=motif_labels,
		)

		transition_table.to_csv(
			os.path.join(
				output_folder,
				prefix
				+ '_transition_probabilities.csv',
			)
		)

		# --------------------------------------------------------
		# Visualizations
		# --------------------------------------------------------

		_plot_timeline(
			time_points,
			states,
			os.path.join(
				output_folder,
				prefix + '_timeline.png',
			),
			animal_id,
		)

		_plot_composition(
			composition,
			behavior_names,
			os.path.join(
				output_folder,
				prefix
				+ '_composition_heatmap.png',
			),
			animal_id,
		)

		# --------------------------------------------------------
		# Save fitted model arrays
		# --------------------------------------------------------

		np.savez_compressed(
			os.path.join(
				output_folder,
				prefix + '_model.npz',
			),
			start_probability=model.startprob_,
			transition_probability=model.transmat_,
			window_means=model.means_,
			window_covariances=model.covars_,
			window_features=features,
			window_centers=centers,
			decoded_window_motif=(
				window_states + 1
			),
			decoded_frame_motif=(
				states + 1
			),
			behavior_composition=composition,
			behavior_names=np.asarray(
				behavior_names,
				dtype=str,
			),
			log_likelihood=np.asarray(
				model.log_likelihood_history_,
				dtype=float,
			),
		)

		# --------------------------------------------------------
		# Metadata
		# --------------------------------------------------------

		try:
			metadata_animal_id = int(animal_id)
		except (TypeError, ValueError):
			metadata_animal_id = str(animal_id)

		metadata = {
			'animal_id': metadata_animal_id,
			'model_type':
				'window_composition_gaussian_hmm',
			'n_motifs': n_motifs,
			'window_size_frames': window_size,
			'window_step_frames': window_step,
			'random_seed': random_seed,
			'maximum_iterations': max_iter,
			'iterations': len(
				model.log_likelihood_history_
			),
			'final_log_likelihood': float(
				model.log_likelihood_history_[-1]
			),
			'na_frames_imputed': int(
				missing_count
			),
			'n_frames': int(
				len(observations)
			),
			'n_windows': int(
				len(features)
			),
			'behavior_names': behavior_names,
		}

		with open(
			os.path.join(
				output_folder,
				prefix + '_metadata.json',
			),
			'w',
			encoding='utf-8',
		) as handle:

			json.dump(
				metadata,
				handle,
			indent=2,
			ensure_ascii=False,
			)

		all_results[animal_id] = {
			'model': model,
			'states': states,
			'window_states': window_states,
			'window_features': features,
			'window_centers': centers,
			'composition': composition,
			'summary': summary,
			'metadata': metadata,
		}

	return all_results


def apply_motif_names(
	output_folder,
	animal_id,
	motif_names,
):
	"""
	Apply user-defined motif names to exported motif results.

	Parameters
	----------
	output_folder : str
		Folder containing motif result files.

	animal_id : int or str
		Animal ID used in exported filenames.

	motif_names : dict
		Mapping from motif ID to user-defined name.

		Example:
		{
			1: 'Exploration',
			2: 'Feeding',
			3: 'Grooming',
			4: 'Resting',
		}
	"""

	output_folder = Path(output_folder)
	prefix = f'behavioral_motifs_ID{animal_id}'

	clean_names = {}

	for motif_id, motif_name in motif_names.items():

		motif_id = int(motif_id)
		motif_name = str(motif_name).strip()

		if not motif_name:
			raise ValueError(
				f'Motif {motif_id} must have a name.'
			)

		clean_names[motif_id] = motif_name

	if len(set(clean_names.values())) != len(clean_names):
		raise ValueError(
			'Every motif name must be unique.'
		)

	# ---------------------------------------------------------
	# Summary table
	# ---------------------------------------------------------

	summary_path = output_folder / f'{prefix}_summary.csv'

	if not summary_path.exists():
		raise FileNotFoundError(
			f'Motif summary file was not found:\n{summary_path}'
		)

	summary = pd.read_csv(summary_path)

	if 'motif_id' not in summary.columns:
		raise ValueError(
			f'{summary_path.name} does not contain motif_id.'
		)

	summary['motif_name'] = (
		summary['motif_id']
		.astype(int)
		.map(clean_names)
	)

	if summary['motif_name'].isna().any():

		missing_ids = (
			summary.loc[
				summary['motif_name'].isna(),
				'motif_id',
			]
			.astype(int)
			.tolist()
		)

		raise ValueError(
			'Missing names for motif IDs: '
			+ ', '.join(map(str, missing_ids))
		)

	if 'common_preceding_motif' in summary.columns:

		summary['common_preceding_motif_name'] = (
			summary['common_preceding_motif']
			.apply(
				lambda value:
				clean_names.get(int(value), '')
				if pd.notna(value)
				else ''
			)
		)

	if 'common_following_motif' in summary.columns:

		summary['common_following_motif_name'] = (
			summary['common_following_motif']
			.apply(
				lambda value:
				clean_names.get(int(value), '')
				if pd.notna(value)
				else ''
			)
		)

	summary.to_csv(
		summary_path,
		index=False,
	)

	# ---------------------------------------------------------
	# Frame assignments
	# ---------------------------------------------------------

	frame_path = (
		output_folder
		/ f'{prefix}_frame_assignments.csv'
	)

	if frame_path.exists():

		frame_table = pd.read_csv(frame_path)

		if 'motif_id' in frame_table.columns:

			frame_table['motif_name'] = (
				frame_table['motif_id']
				.astype(int)
				.map(clean_names)
			)

			frame_table.to_csv(
				frame_path,
				index=False,
			)

	# ---------------------------------------------------------
	# Window assignments
	# ---------------------------------------------------------

	window_path = (
		output_folder
		/ f'{prefix}_window_assignments.csv'
	)

	if window_path.exists():

		window_table = pd.read_csv(window_path)

		if 'motif_id' in window_table.columns:

			window_table['motif_name'] = (
				window_table['motif_id']
				.astype(int)
				.map(clean_names)
			)

			window_table.to_csv(
				window_path,
				index=False,
			)

	# ---------------------------------------------------------
	# Emission/composition table
	# ---------------------------------------------------------

	emission_path = (
		output_folder
		/ f'{prefix}_emission_probabilities.csv'
	)

	if emission_path.exists():

		emission = pd.read_csv(emission_path)

		# Remove an automatically saved pandas index column.
		unnamed_columns = [
			column
			for column in emission.columns
			if str(column).startswith('Unnamed:')
		]

		if unnamed_columns:

			# Preserve the old motif-label column only when no explicit
			# motif_name column is already available.
			if 'motif_name' not in emission.columns:

				emission.rename(
					columns={
						unnamed_columns[0]: 'motif_name',
					},
					inplace=True,
				)

				if len(unnamed_columns) > 1:

					emission.drop(
						columns=unnamed_columns[1:],
						inplace=True,
					)

			else:

				emission.drop(
					columns=unnamed_columns,
					inplace=True,
				)

		# Create or update motif IDs without duplicate insertion.
		expected_ids = np.arange(
			1,
			len(emission) + 1,
			dtype=int,
		)

		if 'motif_id' in emission.columns:

			emission['motif_id'] = expected_ids

		else:

			emission.insert(
				0,
				'motif_id',
				expected_ids,
			)

		# Create or update names without duplicate insertion.
		updated_names = [
			clean_names.get(
				motif_id,
				f'Motif {motif_id}',
			)
			for motif_id in expected_ids
		]

		if 'motif_name' in emission.columns:

			emission['motif_name'] = updated_names

		else:

			emission.insert(
				1,
				'motif_name',
				updated_names,
			)

		# Keep identifying columns at the front.
		other_columns = [
			column
			for column in emission.columns
			if column not in {
				'motif_id',
				'motif_name',
			}
		]

		emission = emission[
			[
				'motif_id',
				'motif_name',
			]
			+ other_columns
		]

		emission.to_csv(
			emission_path,
			index=False,
		)
		
	# ---------------------------------------------------------
	# Transition matrix
	# ---------------------------------------------------------

	transition_path = (
		output_folder
		/ f'{prefix}_transition_probabilities.csv'
	)

	if transition_path.exists():

		transition = pd.read_csv(
			transition_path,
			index_col=0,
		)

		if transition.shape[0] != transition.shape[1]:

			raise ValueError(
				'The motif transition table must be square, but '
				f'its shape is {transition.shape}.'
			)

		ordered_names = [
			clean_names.get(
				motif_id,
				f'Motif {motif_id}',
			)
			for motif_id in range(
				1,
				len(transition) + 1,
			)
		]

		transition.index = ordered_names
		transition.columns = ordered_names

		transition.to_csv(
			transition_path,
			index=True,
		)

	# ---------------------------------------------------------
	# Metadata
	# ---------------------------------------------------------

	metadata_path = (
		output_folder
		/ f'{prefix}_metadata.json'
	)

	if metadata_path.exists():

		with open(
			metadata_path,
			'r',
			encoding='utf-8',
		) as handle:

			metadata = json.load(handle)

	else:

		metadata = {}

	metadata['motif_names'] = {
		str(motif_id): motif_name
		for motif_id, motif_name in clean_names.items()
	}

	with open(
		metadata_path,
		'w',
		encoding='utf-8',
	) as handle:

		json.dump(
			metadata,
			handle,
			indent=2,
			ensure_ascii=False,
		)

	# ---------------------------------------------------------
	# Regenerate timeline with user names
	# ---------------------------------------------------------

	if frame_path.exists():

		frame_table = pd.read_csv(frame_path)

		if {
			'motif_id',
			'motif_name',
		}.issubset(frame_table.columns):

			if 'time_seconds' in frame_table.columns:
				x_values = (
					frame_table['time_seconds']
					.to_numpy(dtype=float)
				)
				x_label = 'Time (seconds)'

			else:
				x_values = np.arange(
					len(frame_table)
				)
				x_label = 'Frame'

			motif_ids = (
				frame_table['motif_id']
				.astype(int)
				.to_numpy()
			)

			figure, axis = plt.subplots(
				figsize=(14, 3.5)
			)

			axis.scatter(
				x_values,
				motif_ids,
				c=motif_ids,
				cmap='viridis',
				marker='s',
				s=8,
				linewidths=0,
			)

			ordered_ids = sorted(clean_names)

			axis.set_yticks(ordered_ids)

			axis.set_yticklabels([
				clean_names[motif_id]
				for motif_id in ordered_ids
			])

			axis.set_xlabel(x_label)
			axis.set_ylabel('Behavioral motif')

			axis.set_title(
				f'Behavioral Motif Timeline — ID {animal_id}'
			)

			axis.grid(
				axis='x',
				alpha=0.2,
			)

			figure.tight_layout()

			figure.savefig(
				output_folder
				/ f'{prefix}_timeline.png',
				dpi=300,
				bbox_inches='tight',
			)

			plt.close(figure)

	# ---------------------------------------------------------
	# Regenerate composition heatmap with user names
	# ---------------------------------------------------------

	if emission_path.exists():

		emission = pd.read_csv(emission_path)

		excluded_columns = {
			'motif_id',
			'motif_name',
		}

		behavior_columns = [
			column
			for column in emission.columns
			if column not in excluded_columns
		]

		if behavior_columns:

			composition = (
				emission[behavior_columns]
				.to_numpy(dtype=float)
			)

			figure_width = max(
				8,
				1.1 * len(behavior_columns),
			)

			figure_height = max(
				4,
				0.7 * len(emission),
			)

			figure, axis = plt.subplots(
				figsize=(
					figure_width,
					figure_height,
				)
			)

			image = axis.imshow(
				composition,
				aspect='auto',
				interpolation='nearest',
				cmap='inferno',
				vmin=0,
				vmax=1,
			)

			axis.set_xticks(
				np.arange(
					len(behavior_columns)
				)
			)

			axis.set_xticklabels(
				behavior_columns,
				rotation=45,
				ha='right',
			)

			axis.set_yticks(
				np.arange(
					len(emission)
				)
			)

			axis.set_yticklabels(
				emission['motif_name']
				.tolist()
			)

			axis.set_xlabel(
				'Observed behavior'
			)

			axis.set_ylabel(
				'Behavioral motif'
			)

			axis.set_title(
				f'Motif Composition — ID {animal_id}'
			)

			colorbar = figure.colorbar(
				image,
				ax=axis,
			)

			colorbar.set_label(
				'Behavior composition'
			)

			figure.tight_layout()

			figure.savefig(
				output_folder
				/ f'{prefix}_composition_heatmap.png',
				dpi=300,
				bbox_inches='tight',
			)

			plt.close(figure)

	return clean_names