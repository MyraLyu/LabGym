'''
Behavioral motif discovery for LabGym.

This module performs post-analysis behavioral motif discovery using
sliding-window behavior compositions and a Gaussian Hidden Markov Model.

The HMM identifies latent temporal states ("behavioral motifs") from
local behavior-composition vectors. User-defined sequence constraints
are applied after HMM inference to individual motif occurrences.

Sequence constraints do NOT alter or refit the HMM.

Outputs include:
- motif summary table
- frame-level motif assignments
- sliding-window motif assignments
- motif behavior composition
- motif transition probabilities
- motif timeline
- motif composition heatmap
- fitted model arrays
- metadata

User-defined motif names may subsequently be assigned with
apply_motif_names().
'''


# ============================================================
# Imports
# ============================================================

from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Numerical utilities
# ============================================================


def _logsumexp(values, axis=None):

	values=np.asarray(
		values,
		dtype=float,
	)

	max_value=np.max(
		values,
		axis=axis,
		keepdims=True,
	)

	max_value=np.where(
		np.isfinite(max_value),
		max_value,
		0.0,
	)

	result=(
		max_value
		+np.log(
			np.sum(
				np.exp(
					values-max_value
				),
				axis=axis,
				keepdims=True,
			)
			+1e-300
		)
	)

	if axis is not None:

		result=np.squeeze(
			result,
			axis=axis,
		)

	return result



def _normalize(vector):

	vector=np.asarray(
		vector,
		dtype=float,
	)

	vector=np.maximum(
		vector,
		1e-12,
	)

	total=vector.sum()

	if total<=0:

		return (
			np.ones_like(vector)
			/len(vector)
		)

	return vector/total



def _normalize_rows(matrix):

	matrix=np.asarray(
		matrix,
		dtype=float,
	)

	matrix=np.maximum(
		matrix,
		1e-12,
	)

	row_sums=matrix.sum(
		axis=1,
		keepdims=True,
	)

	row_sums[
		row_sums<=0
	]=1.0

	return matrix/row_sums


# ============================================================
# Gaussian HMM
# ============================================================


class GaussianHMM:

	'''
	Diagonal-covariance Gaussian Hidden Markov Model.

	The observation at each HMM time step is a sliding-window
	behavior-composition vector.
	'''


	def __init__(
		self,
		n_states,
		n_features,
		random_seed=42,
		max_iter=200,
		tol=1e-4,
		min_covar=1e-4,
	):

		self.n_states=int(
			n_states
		)

		self.n_features=int(
			n_features
		)

		self.random_seed=int(
			random_seed
		)

		self.max_iter=int(
			max_iter
		)

		self.tol=float(
			tol
		)

		self.min_covar=float(
			min_covar
		)

		self.startprob_=None
		self.transmat_=None
		self.means_=None
		self.covars_=None

		self.log_likelihood_history_=[]



	def _initialize(self, X):

		X=np.asarray(
			X,
			dtype=float,
		)

		n_samples=len(X)

		if n_samples<self.n_states:

			raise ValueError(
				'The number of observations must be at least '
				'the number of motifs.'
			)

		rng=np.random.default_rng(
			self.random_seed
		)

		self.startprob_=np.ones(
			self.n_states,
			dtype=float,
		)

		self.startprob_=_normalize(
			self.startprob_
		)

		# Favor persistence during initialization.
		self.transmat_=np.ones(
			(
				self.n_states,
				self.n_states,
			),
			dtype=float,
		)

		np.fill_diagonal(
			self.transmat_,
			8.0,
		)

		self.transmat_=_normalize_rows(
			self.transmat_
		)

		indices=rng.choice(
			n_samples,
			size=self.n_states,
			replace=False,
		)

		self.means_=X[
			indices
		].copy()

		global_variance=np.var(
			X,
			axis=0,
		)

		global_variance=np.maximum(
			global_variance,
			self.min_covar,
		)

		self.covars_=np.tile(
			global_variance,
			(
				self.n_states,
				1,
			),
		)



	def _log_emission_probability(self, X):

		X=np.asarray(
			X,
			dtype=float,
		)

		n_samples=len(X)

		result=np.zeros(
			(
				n_samples,
				self.n_states,
			),
			dtype=float,
		)

		for state in range(
			self.n_states
		):

			variance=np.maximum(
				self.covars_[
					state
				],
				self.min_covar,
			)

			diff=(
				X
				-self.means_[
					state
				]
			)

			result[
				:,
				state,
			]=(
				-0.5
				*(
					self.n_features
					*np.log(
						2.0*np.pi
					)
					+np.sum(
						np.log(
							variance
						)
					)
					+np.sum(
						(diff*diff)
						/variance,
						axis=1,
					)
				)
			)

		return result



	def _forward(self, log_emission):

		n_samples=len(
			log_emission
		)

		log_start=np.log(
			self.startprob_
			+1e-300
		)

		log_transition=np.log(
			self.transmat_
			+1e-300
		)

		alpha=np.zeros(
			(
				n_samples,
				self.n_states,
			),
			dtype=float,
		)

		alpha[0]=(
			log_start
			+log_emission[0]
		)

		for t in range(
			1,
			n_samples,
		):

			alpha[t]=(
				log_emission[t]
				+_logsumexp(
					alpha[
						t-1
					][:,None]
					+log_transition,
					axis=0,
				)
			)

		log_likelihood=float(
			_logsumexp(
				alpha[-1]
			)
		)

		return (
			alpha,
			log_likelihood,
		)



	def _backward(self, log_emission):

		n_samples=len(
			log_emission
		)

		log_transition=np.log(
			self.transmat_
			+1e-300
		)

		beta=np.zeros(
			(
				n_samples,
				self.n_states,
			),
			dtype=float,
		)

		for t in range(
			n_samples-2,
			-1,
			-1,
		):

			beta[t]=_logsumexp(
				log_transition
				+log_emission[
					t+1
				][None,:]
				+beta[
					t+1
				][None,:],
				axis=1,
			)

		return beta



	def fit(self, X):

		X=np.asarray(
			X,
			dtype=float,
		)

		if X.ndim!=2:

			raise ValueError(
				'Gaussian HMM input must be a '
				'2-dimensional matrix.'
			)

		if X.shape[1]!=self.n_features:

			raise ValueError(
				'Gaussian HMM feature dimension does '
				'not match n_features.'
			)

		self._initialize(
			X
		)

		self.log_likelihood_history_=[]

		previous_log_likelihood=None

		for _ in range(
			self.max_iter
		):

			log_emission=(
				self._log_emission_probability(
					X
				)
			)

			alpha,log_likelihood=(
				self._forward(
					log_emission
				)
			)

			beta=self._backward(
				log_emission
			)

			log_gamma=(
				alpha
				+beta
				-log_likelihood
			)

			gamma=np.exp(
				log_gamma
			)

			gamma=(
				gamma
				/np.maximum(
					gamma.sum(
						axis=1,
						keepdims=True,
					),
					1e-300,
				)
			)

			xi_sum=np.zeros(
				(
					self.n_states,
					self.n_states,
				),
				dtype=float,
			)

			log_transition=np.log(
				self.transmat_
				+1e-300
			)

			for t in range(
				len(X)-1
			):

				log_xi=(
					alpha[
						t
					][:,None]
					+log_transition
					+log_emission[
						t+1
					][None,:]
					+beta[
						t+1
					][None,:]
					-log_likelihood
				)

				xi=np.exp(
					log_xi
				)

				xi_sum+=xi

			# Start probabilities.
			self.startprob_=_normalize(
				gamma[0]
				+1e-8
			)

			# Transition probabilities.
			self.transmat_=_normalize_rows(
				xi_sum
				+1e-8
			)

			state_weights=np.maximum(
				gamma.sum(
					axis=0
				),
				1e-8,
			)

			new_means=np.zeros_like(
				self.means_
			)

			new_covars=np.zeros_like(
				self.covars_
			)

			for state in range(
				self.n_states
			):

				weights=gamma[
					:,
					state,
				][:,None]

				new_means[
					state
				]=(
					(
						weights
						*X
					).sum(
						axis=0
					)
					/state_weights[
						state
					]
				)

				diff=(
					X
					-new_means[
						state
					]
				)

				new_covars[
					state
				]=(
					(
						weights
						*(diff*diff)
					).sum(
						axis=0
					)
					/state_weights[
						state
					]
				)

				new_covars[
					state
				]=np.maximum(
					new_covars[
						state
					],
					self.min_covar,
				)

			self.means_=new_means
			self.covars_=new_covars

			self.log_likelihood_history_.append(
				float(
					log_likelihood
				)
			)

			if previous_log_likelihood is not None:

				improvement=(
					log_likelihood
					-previous_log_likelihood
				)

				if abs(
					improvement
				)<self.tol:

					break

			previous_log_likelihood=(
				log_likelihood
			)

		return self



	def predict(self, X):

		'''
		Viterbi decoding.
		'''

		X=np.asarray(
			X,
			dtype=float,
		)

		log_emission=(
			self._log_emission_probability(
				X
			)
		)

		log_start=np.log(
			self.startprob_
			+1e-300
		)

		log_transition=np.log(
			self.transmat_
			+1e-300
		)

		n_samples=len(X)

		score=np.zeros(
			(
				n_samples,
				self.n_states,
			),
			dtype=float,
		)

		backpointer=np.zeros(
			(
				n_samples,
				self.n_states,
			),
			dtype=int,
		)

		score[0]=(
			log_start
			+log_emission[0]
		)

		for t in range(
			1,
			n_samples,
		):

			candidates=(
				score[
					t-1
				][:,None]
				+log_transition
			)

			backpointer[
				t
			]=np.argmax(
				candidates,
				axis=0,
			)

			score[t]=(
				np.max(
					candidates,
					axis=0,
				)
				+log_emission[t]
			)

		states=np.zeros(
			n_samples,
			dtype=int,
		)

		states[-1]=int(
			np.argmax(
				score[-1]
			)
		)

		for t in range(
			n_samples-2,
			-1,
			-1,
		):

			states[t]=(
				backpointer[
					t+1,
					states[
						t+1
					],
				]
			)

		return states


# ============================================================
# Input preparation
# ============================================================


def _prepare_event_sequences(
	event_probability
):

	'''
	Normalize LabGym event input into:

	[
		(animal_id, event_sequence),
		...
	]

	Supported:
	- dict
	- nested list
	- flat list
	'''

	if event_probability is None:

		raise ValueError(
			'No behavioral event data were found.'
		)

	if isinstance(
		event_probability,
		dict,
	):

		if len(
			event_probability
		)==0:

			raise ValueError(
				'No behavioral event data were found.'
			)

		return list(
			event_probability.items()
		)

	if isinstance(
		event_probability,
		(
			list,
			tuple,
			np.ndarray,
		),
	):

		if len(
			event_probability
		)==0:

			raise ValueError(
				'No behavioral event data were found.'
			)

		first=event_probability[0]

		# LabGym single-animal sequence can itself contain
		# [behavior, probability] pairs.
		if isinstance(
			first,
			(
				list,
				tuple,
				np.ndarray,
			),
		):

			if (
				len(first)>=1
				and (
					len(first)==2
					and isinstance(
						first[0],
						(
							str,
							type(None),
						),
					)
				)
			):

				return [
					(
						0,
						list(
							event_probability
						),
					)
				]

			# Otherwise assume nested animal sequences.
			return list(
				enumerate(
					event_probability
				)
			)

		# Flat behavior sequence.
		return [
			(
				0,
				list(
					event_probability
				),
			)
		]

	raise TypeError(
		'Behavior event input must be a dictionary, '
		'list, tuple, or numpy array.'
	)



def _fill_missing_behaviors(events):

	'''
	Convert LabGym event entries to clean behavior labels.

	LabGym entries may look like:

	['sniffing', 0.95]
	['NA', -1]

	Missing values are forward-filled and then backward-filled.
	'''

	events=list(
		events
	)

	cleaned=[]

	for event in events:

		# LabGym native format:
		# [behavior_name, probability]
		if isinstance(
			event,
			(
				list,
				tuple,
				np.ndarray,
			),
		):

			if len(event)==0:

				cleaned.append(
					None
				)

				continue

			behavior=event[0]

		else:

			behavior=event

		if behavior is None:

			cleaned.append(
				None
			)

			continue

		if isinstance(
			behavior,
			(
				float,
				np.floating,
			),
		):

			if np.isnan(
				behavior
			):

				cleaned.append(
					None
				)

				continue

		behavior=str(
			behavior
		).strip()

		if behavior.lower() in {
			'',
			'na',
			'nan',
			'none',
			'null',
		}:

			cleaned.append(
				None
			)

		else:

			cleaned.append(
				behavior
			)

	missing_count=sum(
		value is None
		for value in cleaned
	)

	if missing_count==len(
		cleaned
	):

		raise ValueError(
			'The behavior sequence contains no '
			'valid behavior labels.'
		)

	# Forward fill.
	last_valid=None

	for index,value in enumerate(
		cleaned
	):

		if value is None:

			if last_valid is not None:

				cleaned[
					index
				]=last_valid

		else:

			last_valid=value

	# Backward fill beginning.
	next_valid=None

	for index in range(
		len(cleaned)-1,
		-1,
		-1,
	):

		if cleaned[
			index
		] is None:

			if next_valid is not None:

				cleaned[
					index
				]=next_valid

		else:

			next_valid=cleaned[
				index
			]

	return (
		cleaned,
		missing_count,
	)



def _encode_behaviors(
	labels,
	behavior_names,
):

	behavior_names=[
		str(name)
		for name in behavior_names
	]

	lookup={
		name:index
		for index,name
		in enumerate(
			behavior_names
		)
	}

	lookup_lower={
		name.lower():index
		for index,name
		in enumerate(
			behavior_names
		)
	}

	encoded=[]

	for label in labels:

		label=str(
			label
		)

		if label in lookup:

			encoded.append(
				lookup[
					label
				]
			)

		elif label.lower() in lookup_lower:

			encoded.append(
				lookup_lower[
					label.lower()
				]
			)

		else:

			raise ValueError(
				'Behavior "'
				+label
				+'" is present in the event sequence '
				'but not in the behavior-name list.'
			)

	return np.asarray(
		encoded,
		dtype=int,
	)


# ============================================================
# Sliding-window representation
# ============================================================


def _window_compositions(
	observations,
	n_behaviors,
	window_size,
	window_step,
):

	observations=np.asarray(
		observations,
		dtype=int,
	)

	window_size=int(
		window_size
	)

	window_step=int(
		window_step
	)

	if window_size<2:

		raise ValueError(
			'Window size must contain at least 2 frames.'
		)

	if window_step<1:

		raise ValueError(
			'Window step must be at least 1 frame.'
		)

	if window_size>len(
		observations
	):

		raise ValueError(
			'Window size exceeds the number of '
			'analyzed frames.'
		)

	features=[]
	centers=[]
	bounds=[]

	start=0

	while (
		start+window_size
		<=len(observations)
	):

		end=(
			start
			+window_size
		)

		window=observations[
			start:end
		]

		counts=np.bincount(
			window,
			minlength=n_behaviors,
		).astype(
			float
		)

		composition=(
			counts
			/float(
				window_size
			)
		)

		features.append(
			composition
		)

		centers.append(
			int(
				start
				+(window_size-1)/2
			)
		)

		bounds.append(
			(
				int(start),
				int(end),
			)
		)

		start+=window_step

	# Ensure the end of the video is represented.
	last_start=(
		len(observations)
		-window_size
	)

	if (
		len(bounds)>0
		and bounds[-1][0]!=last_start
	):

		start=last_start
		end=len(
			observations
		)

		window=observations[
			start:end
		]

		counts=np.bincount(
			window,
			minlength=n_behaviors,
		).astype(
			float
		)

		features.append(
			counts
			/float(
				window_size
			)
		)

		centers.append(
			int(
				start
				+(window_size-1)/2
			)
		)

		bounds.append(
			(
				int(start),
				int(end),
			)
		)

	return (
		np.asarray(
			features,
			dtype=float,
		),
		np.asarray(
			centers,
			dtype=int,
		),
		bounds,
	)



def _map_window_states_to_frames(
	window_states,
	bounds,
	n_frames,
	n_motifs,
):

	'''
	Map overlapping window states back to frame-level states.

	Each frame receives the motif with the largest number
	of overlapping-window votes.
	'''

	votes=np.zeros(
		(
			n_frames,
			n_motifs,
		),
		dtype=int,
	)

	for state,bound in zip(
		window_states,
		bounds,
	):

		start,end=bound

		votes[
			start:end,
			int(state),
		]+=1

	states=np.argmax(
		votes,
		axis=1,
	)

	no_votes=np.where(
		votes.sum(
			axis=1
		)==0
	)[0]

	if len(
		no_votes
	)>0:

		covered=np.where(
			votes.sum(
				axis=1
			)>0
		)[0]

		if len(
			covered
		)==0:

			raise ValueError(
				'No frames received motif assignments.'
			)

		for frame in no_votes:

			nearest=covered[
				np.argmin(
					np.abs(
						covered-frame
					)
				)
			]

			states[
				frame
			]=states[
				nearest
			]

	return states.astype(
		int
	)



def _motif_composition(
	features,
	window_states,
	n_motifs,
	n_behaviors,
):

	composition=np.zeros(
		(
			n_motifs,
			n_behaviors,
		),
		dtype=float,
	)

	global_composition=np.mean(
		features,
		axis=0,
	)

	for motif in range(
		n_motifs
	):

		selected=features[
			window_states==motif
		]

		if len(
			selected
		)>0:

			composition[
				motif
			]=np.mean(
				selected,
				axis=0,
			)

		else:

			composition[
				motif
			]=global_composition

		row_sum=composition[
			motif
		].sum()

		if row_sum>0:

			composition[
				motif
			]/=row_sum

	return composition


# ============================================================
# Motif occurrence utilities
# ============================================================


def _contiguous_runs(states):

	states=np.asarray(
		states,
		dtype=int,
	)

	if len(
		states
	)==0:

		return []

	runs=[]

	start=0

	current=int(
		states[0]
	)

	for index in range(
		1,
		len(states),
	):

		state=int(
			states[
				index
			]
		)

		if state!=current:

			runs.append(
				(
					current,
					start,
					index,
				)
			)

			current=state
			start=index

	runs.append(
		(
			current,
			start,
			len(states),
		)
	)

	return runs



def _collapse_repeats(sequence):

	'''
	Example:

	A A A B B C C A

	becomes:

	A B C A
	'''

	result=[]

	for value in sequence:

		if (
			len(result)==0
			or result[-1]!=value
		):

			result.append(
				value
			)

	return result


# ============================================================
# User-defined sequence constraints
# ============================================================


def _validate_motif_sequence(
	sequence,
	min_sequence_length=2,
	max_sequence_length=5,
	sequence_rule='none',
):

	'''
	Validate one collapsed motif occurrence.

	sequence_rule:

	none
		No structural restriction.

	unique
		Every behavior can occur only once.

		A -> B -> C     valid
		A -> B -> A     invalid

	recurrence
		At least one behavior must occur again.

		A -> B -> C     invalid
		A -> B -> A     valid
	'''

	if sequence is None:

		return False

	sequence=[
		behavior
		for behavior in sequence
		if behavior is not None
	]

	if len(
		sequence
	)<int(
		min_sequence_length
	):

		return False

	if (
		max_sequence_length is not None
		and len(sequence)
		>int(
			max_sequence_length
		)
	):

		return False

	if sequence_rule=='none':

		return True

	if sequence_rule=='unique':

		return (
			len(sequence)
			==len(
				set(sequence)
			)
		)

	if sequence_rule=='recurrence':

		return (
			len(sequence)
			>len(
				set(sequence)
			)
		)

	raise ValueError(
		'Unknown sequence rule: '
		+str(
			sequence_rule
		)
	)



def _get_valid_motif_runs(
	labels,
	runs,
	motif_state,
	min_sequence_length=2,
	max_sequence_length=5,
	sequence_rule='none',
):

	valid_runs=[]

	for state,start,end in runs:

		if state!=motif_state:

			continue

		raw_sequence=labels[
			start:end
		]

		collapsed_sequence=(
			_collapse_repeats(
				raw_sequence
			)
		)

		is_valid=(
			_validate_motif_sequence(
				collapsed_sequence,
				min_sequence_length=(
					min_sequence_length
				),
				max_sequence_length=(
					max_sequence_length
				),
				sequence_rule=(
					sequence_rule
				),
			)
		)

		if not is_valid:

			continue

		valid_runs.append(
			{
				'start':int(
					start
				),
				'end':int(
					end
				),
				'sequence':
					collapsed_sequence,
			}
		)

	return valid_runs



def _dominant_sequence_for_motif(
	valid_runs,
	max_length=5,
):

	if len(
		valid_runs
	)==0:

		return ''

	counter=Counter()

	for run in valid_runs:

		sequence=list(
			run[
				'sequence'
			]
		)

		if len(
			sequence
		)==0:

			continue

		if max_length is not None:

			sequence=sequence[
				:int(
					max_length
				)
			]

		counter[
			tuple(
				sequence
			)
		]+=1

	if len(
		counter
	)==0:

		return ''

	best_sequence=(
		counter.most_common(
			1
		)[0][0]
	)

	return ' -> '.join(
		best_sequence
	)



def _valid_occurrence_composition(
	labels,
	valid_runs,
	behavior_names,
	fallback_composition,
):

	'''
	Calculate behavior composition using only valid
	sequence-constrained motif occurrences.

	If no valid occurrence exists, the original HMM
	composition is retained as a fallback for model
	consistency, although the motif is marked as having
	zero valid occurrences.
	'''

	if len(
		valid_runs
	)==0:

		return np.asarray(
			fallback_composition,
			dtype=float,
		)

	behavior_lookup={
		str(name):index
		for index,name
		in enumerate(
			behavior_names
		)
	}

	behavior_lookup_lower={
		str(name).lower():index
		for index,name
		in enumerate(
			behavior_names
		)
	}

	counts=np.zeros(
		len(
			behavior_names
		),
		dtype=float,
	)

	for run in valid_runs:

		start=run[
			'start'
		]

		end=run[
			'end'
		]

		for behavior in labels[
			start:end
		]:

			behavior=str(
				behavior
			)

			if behavior in behavior_lookup:

				index=behavior_lookup[
					behavior
				]

			elif (
				behavior.lower()
				in behavior_lookup_lower
			):

				index=behavior_lookup_lower[
					behavior.lower()
				]

			else:

				continue

			counts[
				index
			]+=1

	total=counts.sum()

	if total<=0:

		return np.asarray(
			fallback_composition,
			dtype=float,
		)

	return counts/total


# ============================================================
# Motif statistics
# ============================================================


def _motif_statistics(
	states,
	labels,
	time_points,
	composition,
	transition_matrix,
	behavior_names,
	top_n_behaviors,
	min_sequence_length=2,
	max_sequence_length=5,
	sequence_rule='none',
):

	'''
	Generate motif-level statistics after applying sequence
	constraints.

	The HMM states are NOT modified.

	Only individual motif occurrences are accepted/rejected
	according to the user-defined sequence rules.
	'''

	states=np.asarray(
		states,
		dtype=int,
	)

	time_points=np.asarray(
		time_points,
		dtype=float,
	)

	n_motifs=composition.shape[
		0
	]

	runs=_contiguous_runs(
		states
	)

	if len(
		time_points
	)>1:

		frame_duration=float(
			np.median(
				np.diff(
					time_points
				)
			)
		)

		if (
			not np.isfinite(
				frame_duration
			)
			or frame_duration<=0
		):

			frame_duration=1.0

	else:

		frame_duration=1.0

	rows=[]

	filtered_composition=np.zeros_like(
		composition,
		dtype=float,
	)

	for motif_state in range(
		n_motifs
	):

		motif_id=(
			motif_state
			+1
		)

		all_motif_runs=[
			{
				'start':int(
					start
				),
				'end':int(
					end
				),
			}
			for state,start,end
			in runs
			if state==motif_state
		]

		total_occurrences=len(
			all_motif_runs
		)

		valid_runs=(
			_get_valid_motif_runs(
				labels,
				runs,
				motif_state,
				min_sequence_length=(
					min_sequence_length
				),
				max_sequence_length=(
					max_sequence_length
				),
				sequence_rule=(
					sequence_rule
				),
			)
		)

		valid_occurrences=len(
			valid_runs
		)

		rejected_occurrences=(
			total_occurrences
			-valid_occurrences
		)

		valid_frame_count=sum(
			run[
				'end'
			]
			-run[
				'start'
			]
			for run in valid_runs
		)

		occupancy=(
			valid_frame_count
			/len(states)
			if len(states)>0
			else 0.0
		)

		durations=[
			(
				run[
					'end'
				]
				-run[
					'start'
				]
			)
			*frame_duration
			for run in valid_runs
		]

		mean_duration=(
			float(
				np.mean(
					durations
				)
			)
			if len(
				durations
			)>0
			else 0.0
		)

		valid_composition=(
			_valid_occurrence_composition(
				labels,
				valid_runs,
				behavior_names,
				composition[
					motif_state
				],
			)
		)

		filtered_composition[
			motif_state
		]=valid_composition

		probabilities=(
			valid_composition
		)

		order=np.argsort(
			probabilities
		)[::-1]

		top_n=min(
			max(
				1,
				int(
					top_n_behaviors
				),
			),
			len(
				behavior_names
			),
		)

		representative=[]

		if valid_occurrences>0:

			for behavior_index in order[
				:top_n
			]:

				probability=float(
					probabilities[
						behavior_index
					]
				)

				if probability<=0:

					continue

				representative.append(
					str(
						behavior_names[
							behavior_index
						]
					)
					+' ('
					+format(
						probability,
						'.3f',
					)
					+')'
				)

		representative_behaviors=(
			'; '.join(
				representative
			)
		)

		dominant_sequence=(
			_dominant_sequence_for_motif(
				valid_runs,
				max_length=(
					max_sequence_length
				),
			)
		)

		# ----------------------------------------------------
		# Neighboring motifs for valid occurrences
		# ----------------------------------------------------

		preceding=[]
		following=[]

		valid_boundaries={
			(
				run[
					'start'
				],
				run[
					'end'
				],
			)
			for run in valid_runs
		}

		for index,run in enumerate(
			runs
		):

			state,start,end=run

			if state!=motif_state:

				continue

			if (
				start,
				end,
			) not in valid_boundaries:

				continue

			if index>0:

				preceding.append(
					runs[
						index-1
					][0]
					+1
				)

			if index<len(
				runs
			)-1:

				following.append(
					runs[
						index+1
					][0]
					+1
				)

		common_preceding=''

		if len(
			preceding
		)>0:

			common_preceding=int(
				Counter(
					preceding
				).most_common(
					1
				)[0][0]
			)

		common_following=''

		if len(
			following
		)>0:

			common_following=int(
				Counter(
					following
				).most_common(
					1
				)[0][0]
			)

		# ----------------------------------------------------
		# Representative intervals
		# ----------------------------------------------------

		interval_candidates=[]

		for run in valid_runs:

			start=run[
				'start'
			]

			end=run[
				'end'
			]

			start_time=float(
				time_points[
					start
				]
			)

			end_index=min(
				end-1,
				len(
					time_points
				)-1,
			)

			end_time=float(
				time_points[
					end_index
				]
			)

			duration=max(
				0.0,
				end_time
				-start_time,
			)

			interval_candidates.append(
				(
					duration,
					start_time,
					end_time,
				)
			)

		interval_candidates.sort(
			key=lambda value:
				value[0],
			reverse=True,
		)

		representative_intervals=(
			'; '.join(
				[
					f'{start_time:.3f}-{end_time:.3f}s'
					for (
						_,
						start_time,
						end_time,
					)
					in interval_candidates[
						:3
					]
				]
			)
		)

		# ----------------------------------------------------
		# HMM transition information
		# ----------------------------------------------------

		transition_row=(
			transition_matrix[
				motif_state
			].copy()
		)

		if len(
			transition_row
		)>1:

			transition_row[
				motif_state
			]=-1

			most_likely_next=(
				int(
					np.argmax(
						transition_row
					)
				)
				+1
			)

		else:

			most_likely_next=''

		validity_rate=(
			valid_occurrences
			/total_occurrences
			if total_occurrences>0
			else 0.0
		)

		rows.append(
			{
				'motif_id':
					motif_id,

				'motif_name':
					'Motif '
					+str(
						motif_id
					),

				'dominant_behavior_sequence':
					dominant_sequence,

				'representative_behaviors':
					representative_behaviors,

				'occupancy':
					float(
						occupancy
					),

				'mean_duration':
					float(
						mean_duration
					),

				'occurrences':
					int(
						valid_occurrences
					),

				'total_hmm_occurrences':
					int(
						total_occurrences
					),

				'rejected_occurrences':
					int(
						rejected_occurrences
					),

				'validity_rate':
					float(
						validity_rate
					),

				'frame_count':
					int(
						valid_frame_count
					),

				'common_preceding_motif':
					common_preceding,

				'common_following_motif':
					common_following,

				'most_likely_next_motif':
					most_likely_next,

				'representative_intervals':
					representative_intervals,

				'sequence_rule':
					str(
						sequence_rule
					),

				'min_sequence_length':
					int(
						min_sequence_length
					),

				'max_sequence_length':
					(
						int(
							max_sequence_length
						)
						if max_sequence_length
						is not None
						else ''
					),
			}
		)

	return (
		pd.DataFrame(
			rows
		),
		filtered_composition,
	)


# ============================================================
# Plotting
# ============================================================


def _plot_timeline(
	time_points,
	states,
	output_path,
	animal_id,
	motif_names=None,
):

	time_points=np.asarray(
		time_points,
		dtype=float,
	)

	states=np.asarray(
		states,
		dtype=int,
	)

	motif_ids=(
		states+1
	)

	unique_ids=sorted(
		np.unique(
			motif_ids
		).tolist()
	)

	figure,axis=plt.subplots(
		figsize=(
			14,
			3.5,
		)
	)

	axis.scatter(
		time_points,
		motif_ids,
		c=motif_ids,
		cmap='viridis',
		s=8,
		marker='s',
		linewidths=0,
	)

	axis.set_yticks(
		unique_ids
	)

	if motif_names is None:

		labels=[
			'Motif '
			+str(
				motif_id
			)
			for motif_id
			in unique_ids
		]

	else:

		labels=[
			motif_names.get(
				motif_id,
				'Motif '
				+str(
					motif_id
				),
			)
			for motif_id
			in unique_ids
		]

	axis.set_yticklabels(
		labels
	)

	axis.set_xlabel(
		'Time (seconds)'
	)

	axis.set_ylabel(
		'Behavioral motif'
	)

	axis.set_title(
		'Behavioral Motif Timeline - ID '
		+str(
			animal_id
		)
	)

	axis.grid(
		axis='x',
		alpha=0.2,
	)

	figure.tight_layout()

	figure.savefig(
		output_path,
		dpi=300,
		bbox_inches='tight',
	)

	plt.close(
		figure
	)



def _plot_composition(
	composition,
	behavior_names,
	output_path,
	animal_id,
	motif_names=None,
):

	composition=np.asarray(
		composition,
		dtype=float,
	)

	n_motifs=len(
		composition
	)

	figure_width=max(
		8,
		len(
			behavior_names
		)*1.1,
	)

	figure_height=max(
		4,
		n_motifs*0.7,
	)

	figure,axis=plt.subplots(
		figsize=(
			figure_width,
			figure_height,
		)
	)

	image=axis.imshow(
		composition,
		aspect='auto',
		interpolation='nearest',
		cmap='inferno',
		vmin=0,
		vmax=1,
	)

	axis.set_xticks(
		np.arange(
			len(
				behavior_names
			)
		)
	)

	axis.set_xticklabels(
		behavior_names,
		rotation=45,
		ha='right',
	)

	axis.set_yticks(
		np.arange(
			n_motifs
		)
	)

	if motif_names is None:

		y_labels=[
			'Motif '
			+str(
				motif_id
			)
			for motif_id
			in range(
				1,
				n_motifs+1,
			)
		]

	else:

		y_labels=[
			motif_names.get(
				motif_id,
				'Motif '
				+str(
					motif_id
				),
			)
			for motif_id
			in range(
				1,
				n_motifs+1,
			)
		]

	axis.set_yticklabels(
		y_labels
	)

	axis.set_xlabel(
		'Observed behavior'
	)

	axis.set_ylabel(
		'Behavioral motif'
	)

	axis.set_title(
		'Motif Composition - ID '
		+str(
			animal_id
		)
	)

	colorbar=figure.colorbar(
		image,
		ax=axis,
	)

	colorbar.set_label(
		'Behavior composition'
	)

	figure.tight_layout()

	figure.savefig(
		output_path,
		dpi=300,
		bbox_inches='tight',
	)

	plt.close(
		figure
	)


# ============================================================
# Main motif discovery
# ============================================================


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
	min_sequence_length=2,
	max_sequence_length=5,
	sequence_rule='none',
):

	'''
	Discover behavioral motifs independently for each animal.

	Sequence constraints are applied AFTER HMM fitting.

	Parameters
	----------
	sequence_rule : str
		'none'
		'unique'
		'recurrence'
	'''

	event_sequences=(
		_prepare_event_sequences(
			event_probability
		)
	)

	time_points=np.asarray(
		time_points,
		dtype=float,
	)

	if len(
		time_points
	)<2:

		raise ValueError(
			'At least two analyzed frames are required.'
		)

	behavior_names=[
		str(name)
		for name in behavior_names
	]

	if len(
		behavior_names
	)<2:

		raise ValueError(
			'At least two behavior classes are required.'
		)

	n_motifs=int(
		n_motifs
	)

	random_seed=int(
		random_seed
	)

	max_iter=int(
		max_iter
	)

	top_n_behaviors=int(
		top_n_behaviors
	)

	window_size=int(
		window_size
	)

	window_step=int(
		window_step
	)

	min_sequence_length=int(
		min_sequence_length
	)

	if max_sequence_length is not None:

		max_sequence_length=int(
			max_sequence_length
		)

	if n_motifs<2:

		raise ValueError(
			'The number of motifs must be at least 2.'
		)

	if window_size<2:

		raise ValueError(
			'Window size must be at least 2 frames.'
		)

	if window_step<1:

		raise ValueError(
			'Window step must be at least 1 frame.'
		)

	if window_size>len(
		time_points
	):

		raise ValueError(
			'Window size cannot exceed the number '
			'of analyzed frames.'
		)

	if min_sequence_length<1:

		raise ValueError(
			'Minimum sequence length must be at least 1.'
		)

	if (
		max_sequence_length is not None
		and max_sequence_length
		<min_sequence_length
	):

		raise ValueError(
			'Maximum sequence length must be greater '
			'than or equal to minimum sequence length.'
		)

	if sequence_rule not in {
		'none',
		'unique',
		'recurrence',
	}:

		raise ValueError(
			'Sequence rule must be one of: '
			'none, unique, recurrence.'
		)

	output_folder=Path(
		output_folder
	)

	output_folder.mkdir(
		parents=True,
		exist_ok=True,
	)

	all_results={}

	for animal_id,events in event_sequences:

		events=list(
			events
		)

		if len(
			events
		)!=len(
			time_points
		):

			raise ValueError(
				'ID '
				+str(
					animal_id
				)
				+': behavior sequence length ('
				+str(
					len(
						events
					)
				)
				+') does not match time-point length ('
				+str(
					len(
						time_points
					)
				)
				+').'
			)

		labels,missing_count=(
			_fill_missing_behaviors(
				events
			)
		)

		observations=(
			_encode_behaviors(
				labels,
				behavior_names,
			)
		)

		(
			window_features,
			window_centers,
			window_bounds,
		)=_window_compositions(
			observations,
			len(
				behavior_names
			),
			window_size,
			window_step,
		)

		if len(
			window_features
		)<n_motifs:

			raise ValueError(
				'ID '
				+str(
					animal_id
				)
				+': requested '
				+str(
					n_motifs
				)
				+' motifs, but only '
				+str(
					len(
						window_features
					)
				)
				+' sliding windows were generated.'
			)

		model=GaussianHMM(
			n_states=n_motifs,
			n_features=len(
				behavior_names
			),
			random_seed=random_seed,
			max_iter=max_iter,
			tol=tol,
		)

		model.fit(
			window_features
		)

		window_states=model.predict(
			window_features
		)

		frame_states=(
			_map_window_states_to_frames(
				window_states,
				window_bounds,
				len(
					observations
				),
				n_motifs,
			)
		)

		original_composition=(
			_motif_composition(
				window_features,
				window_states,
				n_motifs,
				len(
					behavior_names
				),
			)
		)

		(
			summary,
			filtered_composition,
		)=_motif_statistics(
			frame_states,
			labels,
			time_points,
			original_composition,
			model.transmat_,
			behavior_names,
			max(
				1,
				top_n_behaviors,
			),
			min_sequence_length=(
				min_sequence_length
			),
			max_sequence_length=(
				max_sequence_length
			),
			sequence_rule=(
				sequence_rule
			),
		)

		prefix=(
			'behavioral_motifs_ID'
			+str(
				animal_id
			)
		)

		# ----------------------------------------------------
		# Frame assignments
		# ----------------------------------------------------

		frame_table=pd.DataFrame(
			{
				'frame':
					np.arange(
						len(
							frame_states
						),
						dtype=int,
					),

				'time_seconds':
					time_points,

				'observed_behavior':
					labels,

				'motif_id':
					frame_states+1,

				'motif_name':
					[
						'Motif '
						+str(
							state+1
						)
						for state
						in frame_states
					],
			}
		)

		frame_table.to_csv(
			output_folder
			/(
				prefix
				+'_frame_assignments.csv'
			),
			index=False,
		)

		# ----------------------------------------------------
		# Window assignments
		# ----------------------------------------------------

		window_table=pd.DataFrame(
			window_features,
			columns=behavior_names,
		)

		window_table.insert(
			0,
			'window_end_frame',
			[
				end-1
				for _,end
				in window_bounds
			],
		)

		window_table.insert(
			0,
			'window_start_frame',
			[
				start
				for start,_
				in window_bounds
			],
		)

		window_table.insert(
			0,
			'window_center_frame',
			window_centers,
		)

		window_table[
			'motif_id'
		]=(
			window_states+1
		)

		window_table[
			'motif_name'
		]=[
			'Motif '
			+str(
				state+1
			)
			for state
			in window_states
		]

		window_table.to_csv(
			output_folder
			/(
				prefix
				+'_window_assignments.csv'
			),
			index=False,
		)

		# ----------------------------------------------------
		# Summary
		# ----------------------------------------------------

		summary.to_csv(
			output_folder
			/(
				prefix
				+'_summary.csv'
			),
			index=False,
		)

		# ----------------------------------------------------
		# Filtered motif composition
		# ----------------------------------------------------

		emission_table=pd.DataFrame(
			filtered_composition,
			columns=behavior_names,
		)

		emission_table.insert(
			0,
			'motif_name',
			[
				'Motif '
				+str(
					motif_id
				)
				for motif_id
				in range(
					1,
					n_motifs+1,
				)
			],
		)

		emission_table.insert(
			0,
			'motif_id',
			np.arange(
				1,
				n_motifs+1,
				dtype=int,
			),
		)

		emission_table.to_csv(
			output_folder
			/(
				prefix
				+'_emission_probabilities.csv'
			),
			index=False,
		)

		# ----------------------------------------------------
		# Original HMM composition
		# ----------------------------------------------------

		original_table=pd.DataFrame(
			original_composition,
			columns=behavior_names,
		)

		original_table.insert(
			0,
			'motif_id',
			np.arange(
				1,
				n_motifs+1,
				dtype=int,
			),
		)

		original_table.to_csv(
			output_folder
			/(
				prefix
				+'_original_composition.csv'
			),
			index=False,
		)

		# ----------------------------------------------------
		# Transition probabilities
		# ----------------------------------------------------

		transition_labels=[
			'Motif '
			+str(
				motif_id
			)
			for motif_id
			in range(
				1,
				n_motifs+1,
			)
		]

		transition_table=pd.DataFrame(
			model.transmat_,
			index=transition_labels,
			columns=transition_labels,
		)

		transition_table.to_csv(
			output_folder
			/(
				prefix
				+'_transition_probabilities.csv'
			),
			index=True,
		)

		# ----------------------------------------------------
		# Plots
		# ----------------------------------------------------

		_plot_timeline(
			time_points,
			frame_states,
			output_folder
			/(
				prefix
				+'_timeline.png'
			),
			animal_id,
		)

		_plot_composition(
			filtered_composition,
			behavior_names,
			output_folder
			/(
				prefix
				+'_composition_heatmap.png'
			),
			animal_id,
		)

		# ----------------------------------------------------
		# Model
		# ----------------------------------------------------

		np.savez_compressed(
			output_folder
			/(
				prefix
				+'_model.npz'
			),

			start_probability=
				model.startprob_,

			transition_probability=
				model.transmat_,

			window_means=
				model.means_,

			window_covariances=
				model.covars_,

			window_features=
				window_features,

			window_centers=
				window_centers,

			decoded_window_motif=
				window_states+1,

			decoded_frame_motif=
				frame_states+1,

			original_behavior_composition=
				original_composition,

			filtered_behavior_composition=
				filtered_composition,

			behavior_names=
				np.asarray(
					behavior_names,
					dtype=str,
				),

			log_likelihood=
				np.asarray(
					model.log_likelihood_history_,
					dtype=float,
				),
		)

		# ----------------------------------------------------
		# Metadata
		# ----------------------------------------------------

		try:

			metadata_animal_id=int(
				animal_id
			)

		except (
			TypeError,
			ValueError,
		):

			metadata_animal_id=str(
				animal_id
			)

		metadata={
			'animal_id':
				metadata_animal_id,

			'model_type':
				'window_composition_gaussian_hmm',

			'n_motifs':
				n_motifs,

			'window_size_frames':
				window_size,

			'window_step_frames':
				window_step,

			'random_seed':
				random_seed,

			'maximum_iterations':
				max_iter,

			'iterations':
				len(
					model.log_likelihood_history_
				),

			'final_log_likelihood':
				(
					float(
						model.log_likelihood_history_[
							-1
						]
					)
					if len(
						model.log_likelihood_history_
					)>0
					else None
				),

			'n_frames':
				int(
					len(
						frame_states
					)
				),

			'n_windows':
				int(
					len(
						window_features
					)
				),

			'missing_frames_imputed':
				int(
					missing_count
				),

			'behavior_names':
				behavior_names,

			'sequence_constraints':{
				'min_sequence_length':
					int(
						min_sequence_length
					),

				'max_sequence_length':
					(
						int(
							max_sequence_length
						)
						if max_sequence_length
						is not None
						else None
					),

				'sequence_rule':
					str(
						sequence_rule
					),
			},
		}

		with open(
			output_folder
			/(
				prefix
				+'_metadata.json'
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

		all_results[
			animal_id
		]={
			'model':
				model,

			'states':
				frame_states,

			'window_states':
				window_states,

			'window_features':
				window_features,

			'window_centers':
				window_centers,

			'window_bounds':
				window_bounds,

			'original_composition':
				original_composition,

			'composition':
				filtered_composition,

			'summary':
				summary,

			'metadata':
				metadata,
		}

	return all_results


# ============================================================
# Apply user-defined motif names
# ============================================================


def apply_motif_names(
	output_folder,
	animal_id,
	motif_names,
):

	'''
	Apply user-defined interpretation names to motif outputs.

	This function does NOT alter or refit the HMM.

	Example
	-------
	{
		1: 'Exploration',
		2: 'Feeding',
		3: 'Grooming',
	}
	'''

	output_folder=Path(
		output_folder
	)

	prefix=(
		'behavioral_motifs_ID'
		+str(
			animal_id
		)
	)

	clean_names={}

	for motif_id,motif_name in motif_names.items():

		motif_id=int(
			motif_id
		)

		motif_name=str(
			motif_name
		).strip()

		if len(
			motif_name
		)==0:

			raise ValueError(
				'Motif '
				+str(
					motif_id
				)
				+' must have a name.'
			)

		clean_names[
			motif_id
		]=motif_name

	if len(
		set(
			name.lower()
			for name
			in clean_names.values()
		)
	)!=len(
		clean_names
	):

		raise ValueError(
			'Every motif must have a unique name.'
		)

	# --------------------------------------------------------
	# Summary
	# --------------------------------------------------------

	summary_path=(
		output_folder
		/(
			prefix
			+'_summary.csv'
		)
	)

	if not summary_path.exists():

		raise FileNotFoundError(
			'Motif summary file was not found:\n'
			+str(
				summary_path
			)
		)

	summary=pd.read_csv(
		summary_path
	)

	if 'motif_id' not in summary.columns:

		raise ValueError(
			'The motif summary does not contain motif_id.'
		)

	summary[
		'motif_name'
	]=(
		summary[
			'motif_id'
		]
		.astype(
			int
		)
		.map(
			clean_names
		)
	)

	if summary[
		'motif_name'
	].isna().any():

		missing_ids=(
			summary.loc[
				summary[
					'motif_name'
				].isna(),
				'motif_id',
			]
			.astype(
				int
			)
			.tolist()
		)

		raise ValueError(
			'Missing user-defined names for motif IDs: '
			+', '.join(
				map(
					str,
					missing_ids,
				)
			)
		)

	def _map_optional_motif_name(value):

		if pd.isna(
			value
		):

			return ''

		try:

			return clean_names.get(
				int(
					float(
						value
					)
				),
				'',
			)

		except (
			TypeError,
			ValueError,
		):

			return ''

	if (
		'common_preceding_motif'
		in summary.columns
	):

		summary[
			'common_preceding_motif_name'
		]=summary[
			'common_preceding_motif'
		].apply(
			_map_optional_motif_name
		)

	if (
		'common_following_motif'
		in summary.columns
	):

		summary[
			'common_following_motif_name'
		]=summary[
			'common_following_motif'
		].apply(
			_map_optional_motif_name
		)

	if (
		'most_likely_next_motif'
		in summary.columns
	):

		summary[
			'most_likely_next_motif_name'
		]=summary[
			'most_likely_next_motif'
		].apply(
			_map_optional_motif_name
		)

	summary.to_csv(
		summary_path,
		index=False,
	)

	# --------------------------------------------------------
	# Frame assignments
	# --------------------------------------------------------

	frame_path=(
		output_folder
		/(
			prefix
			+'_frame_assignments.csv'
		)
	)

	if frame_path.exists():

		frame_table=pd.read_csv(
			frame_path
		)

		if (
			'motif_id'
			in frame_table.columns
		):

			frame_table[
				'motif_name'
			]=(
				frame_table[
					'motif_id'
				]
				.astype(
					int
				)
				.map(
					clean_names
				)
			)

			frame_table.to_csv(
				frame_path,
				index=False,
			)

	# --------------------------------------------------------
	# Window assignments
	# --------------------------------------------------------

	window_path=(
		output_folder
		/(
			prefix
			+'_window_assignments.csv'
		)
	)

	if window_path.exists():

		window_table=pd.read_csv(
			window_path
		)

		if (
			'motif_id'
			in window_table.columns
		):

			window_table[
				'motif_name'
			]=(
				window_table[
					'motif_id'
				]
				.astype(
					int
				)
				.map(
					clean_names
				)
			)

			window_table.to_csv(
				window_path,
				index=False,
			)

	# --------------------------------------------------------
	# Composition / emission table
	# --------------------------------------------------------

	emission_path=(
		output_folder
		/(
			prefix
			+'_emission_probabilities.csv'
		)
	)

	if emission_path.exists():

		emission=pd.read_csv(
			emission_path
		)

		unnamed_columns=[
			column
			for column
			in emission.columns
			if str(
				column
			).startswith(
				'Unnamed:'
			)
		]

		if len(
			unnamed_columns
		)>0:

			emission.drop(
				columns=unnamed_columns,
				inplace=True,
			)

		if 'motif_id' not in emission.columns:

			emission.insert(
				0,
				'motif_id',
				np.arange(
					1,
					len(
						emission
					)+1,
					dtype=int,
				),
			)

		emission[
			'motif_id'
		]=np.arange(
			1,
			len(
				emission
			)+1,
			dtype=int,
		)

		updated_names=[
			clean_names.get(
				int(
					motif_id
				),
				'Motif '
				+str(
					motif_id
				),
			)
			for motif_id
			in emission[
				'motif_id'
			]
		]

		if (
			'motif_name'
			in emission.columns
		):

			emission[
				'motif_name'
			]=updated_names

		else:

			emission.insert(
				1,
				'motif_name',
				updated_names,
			)

		other_columns=[
			column
			for column
			in emission.columns
			if column not in {
				'motif_id',
				'motif_name',
			}
		]

		emission=emission[
			[
				'motif_id',
				'motif_name',
			]
			+other_columns
		]

		emission.to_csv(
			emission_path,
			index=False,
		)

	# --------------------------------------------------------
	# Transition matrix
	# --------------------------------------------------------

	transition_path=(
		output_folder
		/(
			prefix
			+'_transition_probabilities.csv'
		)
	)

	if transition_path.exists():

		transition=pd.read_csv(
			transition_path,
			index_col=0,
		)

		if (
			transition.shape[0]
			!=transition.shape[1]
		):

			raise ValueError(
				'Motif transition probability table '
				'must be square.'
			)

		ordered_names=[
			clean_names.get(
				motif_id,
				'Motif '
				+str(
					motif_id
				),
			)
			for motif_id
			in range(
				1,
				len(
					transition
				)+1,
			)
		]

		transition.index=(
			ordered_names
		)

		transition.columns=(
			ordered_names
		)

		transition.to_csv(
			transition_path,
			index=True,
		)

	# --------------------------------------------------------
	# Metadata
	# --------------------------------------------------------

	metadata_path=(
		output_folder
		/(
			prefix
			+'_metadata.json'
		)
	)

	if metadata_path.exists():

		with open(
			metadata_path,
			'r',
			encoding='utf-8',
		) as handle:

			metadata=json.load(
				handle
			)

	else:

		metadata={}

	metadata[
		'motif_names'
	]={
		str(
			motif_id
		):
			motif_name

		for motif_id,motif_name
		in clean_names.items()
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

	# --------------------------------------------------------
	# Regenerate timeline with names
	# --------------------------------------------------------

	if frame_path.exists():

		frame_table=pd.read_csv(
			frame_path
		)

		if {
			'motif_id',
			'motif_name',
		}.issubset(
			frame_table.columns
		):

			if (
				'time_seconds'
				in frame_table.columns
			):

				time_values=(
					frame_table[
						'time_seconds'
					]
					.to_numpy(
						dtype=float
					)
				)

			else:

				time_values=np.arange(
					len(
						frame_table
					),
					dtype=float,
				)

			frame_states=(
				frame_table[
					'motif_id'
				]
				.astype(
					int
				)
				.to_numpy()
				-1
			)

			_plot_timeline(
				time_values,
				frame_states,
				output_folder
				/(
					prefix
					+'_timeline.png'
				),
				animal_id,
				motif_names=(
					clean_names
				),
			)

	# --------------------------------------------------------
	# Regenerate composition heatmap with names
	# --------------------------------------------------------

	if emission_path.exists():

		emission=pd.read_csv(
			emission_path
		)

		behavior_columns=[
			column
			for column
			in emission.columns
			if column not in {
				'motif_id',
				'motif_name',
			}
		]

		if len(
			behavior_columns
		)>0:

			composition=(
				emission[
					behavior_columns
				]
				.to_numpy(
					dtype=float
				)
			)

			_plot_composition(
				composition,
				behavior_columns,
				output_folder
				/(
					prefix
					+'_composition_heatmap.png'
				),
				animal_id,
				motif_names=(
					clean_names
				),
			)

	return clean_names