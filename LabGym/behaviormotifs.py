'''
Behavioral motif discovery for LabGym.

This module performs post-analysis behavioral motif discovery using
sliding-window behavior compositions and a Gaussian Hidden Markov Model.

The fitted latent states are interpreted as behavioral motifs.

Outputs include:
- motif summary table
- frame-level motif assignments
- sliding-window motif assignments
- motif composition probabilities
- motif transition probabilities
- motif timeline
- motif composition heatmap
- fitted model arrays
- metadata

Motif names may subsequently be assigned by the user with
apply_motif_names(). User-defined names do not alter the fitted model.
'''


# Standard library imports.
from collections import Counter
import json
import math
import os
from pathlib import Path


# Related third party imports.
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------


def _logsumexp(values, axis=None):
	'''
	Numerically stable log(sum(exp(values))).
	'''

	values=np.asarray(values,dtype=float)

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
				np.exp(values-max_value),
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
	'''
	Normalize a vector so that it sums to one.
	'''

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
		return np.ones_like(vector)/len(vector)

	return vector/total



def _normalize_rows(matrix):
	'''
	Normalize every matrix row so that it sums to one.
	'''

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

	row_sums[row_sums<=0]=1.0

	return matrix/row_sums


# ---------------------------------------------------------------------
# Gaussian HMM
# ---------------------------------------------------------------------


class GaussianHMM:
	'''
	Simple diagonal-covariance Gaussian Hidden Markov Model.

	This implementation avoids an additional hmmlearn dependency and is
	sufficient for low-dimensional sliding-window behavior compositions.
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

		self.n_states=int(n_states)
		self.n_features=int(n_features)

		self.random_seed=int(random_seed)
		self.max_iter=int(max_iter)
		self.tol=float(tol)
		self.min_covar=float(min_covar)

		self.startprob_=None
		self.transmat_=None
		self.means_=None
		self.covars_=None

		self.log_likelihood_history_=[]



	def _initialize(self,X):
		'''
		Initialize HMM parameters reproducibly.
		'''

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

		# Initialize with strong self-transition probabilities so that
		# inferred motifs naturally persist over consecutive windows.
		self.transmat_=np.full(
			(
				self.n_states,
				self.n_states,
			),
			1.0,
			dtype=float,
		)

		np.fill_diagonal(
			self.transmat_,
			8.0,
		)

		self.transmat_=_normalize_rows(
			self.transmat_
		)

		# Choose reproducible observations as initial state means.
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



	def _log_emission_probability(self,X):
		'''
		Return log p(x_t | z_t=m) for every frame/window and motif.
		'''

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

		for state in range(self.n_states):

			variance=np.maximum(
				self.covars_[state],
				self.min_covar,
			)

			diff=X-self.means_[state]

			result[:,state]=(
				-0.5
				*(
					self.n_features*np.log(
						2.0*np.pi
					)
					+np.sum(
						np.log(variance)
					)
					+np.sum(
						(diff*diff)/variance,
						axis=1,
					)
				)
			)

		return result



	def _forward(self,log_emission):
		'''
		Log-space forward algorithm.
		'''

		n_samples=len(log_emission)

		log_start=np.log(
			self.startprob_+1e-300
		)

		log_transition=np.log(
			self.transmat_+1e-300
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
					alpha[t-1][:,None]
					+log_transition,
					axis=0,
				)
			)

		log_likelihood=float(
			_logsumexp(
				alpha[-1]
			)
		)

		return alpha,log_likelihood



	def _backward(self,log_emission):
		'''
		Log-space backward algorithm.
		'''

		n_samples=len(log_emission)

		log_transition=np.log(
			self.transmat_+1e-300
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
				+log_emission[t+1][None,:]
				+beta[t+1][None,:],
				axis=1,
			)

		return beta



	def fit(self,X):
		'''
		Fit the HMM using Baum-Welch expectation maximization.
		'''

		X=np.asarray(
			X,
			dtype=float,
		)

		if X.ndim!=2:

			raise ValueError(
				'Gaussian HMM input must be a 2-dimensional matrix.'
			)

		if X.shape[1]!=self.n_features:

			raise ValueError(
				'Gaussian HMM feature dimension does not match '
				'the requested feature dimension.'
			)

		self._initialize(X)

		self.log_likelihood_history_=[]

		previous_log_likelihood=None

		for _ in range(
			self.max_iter
		):

			log_emission=(
				self._log_emission_probability(X)
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

			gamma=gamma/np.maximum(
				gamma.sum(
					axis=1,
					keepdims=True,
				),
				1e-300,
			)

			xi_sum=np.zeros(
				(
					self.n_states,
					self.n_states,
				),
				dtype=float,
			)

			log_transition=np.log(
				self.transmat_+1e-300
			)

			for t in range(
				len(X)-1
			):

				log_xi=(
					alpha[t][:,None]
					+log_transition
					+log_emission[t+1][None,:]
					+beta[t+1][None,:]
					-log_likelihood
				)

				xi=np.exp(
					log_xi
				)

				xi_sum+=xi

			# Update start probabilities.
			self.startprob_=_normalize(
				gamma[0]+1e-8
			)

			# Update transition probabilities.
			self.transmat_=_normalize_rows(
				xi_sum+1e-8
			)

			# Update Gaussian parameters.
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

				new_means[state]=(
					(weights*X).sum(
						axis=0
					)
					/state_weights[state]
				)

				diff=(
					X
					-new_means[state]
				)

				new_covars[state]=(
					(
						weights
						*(diff*diff)
					).sum(
						axis=0
					)
					/state_weights[state]
				)

				new_covars[state]=np.maximum(
					new_covars[state],
					self.min_covar,
				)

			self.means_=new_means
			self.covars_=new_covars

			self.log_likelihood_history_.append(
				float(log_likelihood)
			)

			if previous_log_likelihood is not None:

				improvement=(
					log_likelihood
					-previous_log_likelihood
				)

				if abs(improvement)<self.tol:
					break

			previous_log_likelihood=(
				log_likelihood
			)

		return self



	def predict(self,X):
		'''
		Decode the most likely state sequence with Viterbi.
		'''

		X=np.asarray(
			X,
			dtype=float,
		)

		log_emission=(
			self._log_emission_probability(X)
		)

		log_start=np.log(
			self.startprob_+1e-300
		)

		log_transition=np.log(
			self.transmat_+1e-300
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
				score[t-1][:,None]
				+log_transition
			)

			backpointer[t]=np.argmax(
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

			states[t]=backpointer[
				t+1,
				states[t+1],
			]

		return states


# ---------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------


def _prepare_event_sequences(event_probability):
	'''
	Normalize LabGym event input into a list of (animal_id, events).

	Supported structures:

	{
		0: [...],
		1: [...],
	}

	[
		[...],
		[...],
	]

	or a flat sequence:

	[
		'walking',
		'walking',
		'sniffing',
		...
	]
	'''

	if event_probability is None:

		raise ValueError(
			'No behavioral event data were found.'
		)

	if isinstance(
		event_probability,
		dict,
	):

		if len(event_probability)==0:

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

		if len(event_probability)==0:

			raise ValueError(
				'No behavioral event data were found.'
			)

		first=event_probability[0]

		# Flat list = one animal.
		if (
			first is None
			or isinstance(
				first,
				(
					str,
					int,
					float,
					np.integer,
					np.floating,
				),
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

		# Nested list = multiple animals.
		return list(
			enumerate(
				event_probability
			)
		)

	raise TypeError(
		'Behavior event input must be a dictionary, list, '
		'tuple, or numpy array; received '
		+type(event_probability).__name__
		+'.'
	)



def _fill_missing_behaviors(events):
	'''
	Fill missing / NA behavior labels.

	Missing labels are first forward-filled and then backward-filled.
	If an entire sequence is missing, an error is raised.
	'''

	events=list(events)

	cleaned=[]

	for event in events:

		if event is None:

			cleaned.append(None)
			continue

		if isinstance(
			event,
			float,
		):

			if np.isnan(event):

				cleaned.append(None)
				continue

		value=str(event).strip()

		if value.lower() in {
			'',
			'na',
			'nan',
			'none',
			'null',
		}:

			cleaned.append(None)

		else:

			cleaned.append(value)

	missing_count=sum(
		value is None
		for value in cleaned
	)

	if missing_count==len(cleaned):

		raise ValueError(
			'The behavior sequence contains no valid behavior labels.'
		)

	# Forward fill.
	last_valid=None

	for index,value in enumerate(
		cleaned
	):

		if value is None:

			if last_valid is not None:
				cleaned[index]=last_valid

		else:

			last_valid=value

	# Backward fill beginning.
	next_valid=None

	for index in range(
		len(cleaned)-1,
		-1,
		-1,
	):

		if cleaned[index] is None:

			if next_valid is not None:
				cleaned[index]=next_valid

		else:

			next_valid=cleaned[index]

	return cleaned,missing_count



def _encode_behaviors(
	labels,
	behavior_names,
):
	'''
	Convert behavior labels into integer class indices.
	'''

	behavior_names=[
		str(name)
		for name in behavior_names
	]

	lookup={
		name:index
		for index,name in enumerate(
			behavior_names
		)
	}

	# Case-insensitive backup mapping.
	lookup_lower={
		name.lower():index
		for index,name in enumerate(
			behavior_names
		)
	}

	encoded=[]

	for label in labels:

		label=str(label)

		if label in lookup:

			encoded.append(
				lookup[label]
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
				+'" is present in the event sequence but not '
				'in the behavior-name list.'
			)

	return np.asarray(
		encoded,
		dtype=int,
	)


# ---------------------------------------------------------------------
# Sliding-window features
# ---------------------------------------------------------------------


def _window_compositions(
	observations,
	n_behaviors,
	window_size,
	window_step,
):
	'''
	Convert frame-level behavior labels into sliding-window
	behavior-composition vectors.
	'''

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
			'Window size exceeds the number of analyzed frames.'
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
		).astype(float)

		composition=(
			counts
			/float(window_size)
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

		# End is exclusive internally.
		bounds.append(
			(
				int(start),
				int(end),
			)
		)

		start+=window_step

	# Make sure the end of the sequence is represented.
	last_start=(
		len(observations)
		-window_size
	)

	if (
		len(bounds)>0
		and bounds[-1][0]!=last_start
	):

		start=last_start
		end=len(observations)

		window=observations[
			start:end
		]

		counts=np.bincount(
			window,
			minlength=n_behaviors,
		).astype(float)

		features.append(
			counts/float(
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
	Map overlapping sliding-window motif assignments back to frames.

	Each frame receives the motif with the largest number of covering
	windows. This is preferable to simply assigning the nearest window
	center because overlap provides temporal smoothing.
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

	if len(no_votes)>0:

		covered=np.where(
			votes.sum(
				axis=1
			)>0
		)[0]

		if len(covered)==0:

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

			states[frame]=states[
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
	'''
	Average behavior composition of all windows assigned to each motif.
	'''

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

		if len(selected)>0:

			composition[motif]=np.mean(
				selected,
				axis=0,
			)

		else:

			composition[motif]=(
				global_composition
			)

		row_sum=composition[
			motif
		].sum()

		if row_sum>0:

			composition[
				motif
			]/=row_sum

	return composition


# ---------------------------------------------------------------------
# Motif summary statistics
# ---------------------------------------------------------------------


def _contiguous_runs(states):
	'''
	Return contiguous runs as (state, start_frame, end_frame_exclusive).
	'''

	states=np.asarray(
		states,
		dtype=int,
	)

	if len(states)==0:
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
			states[index]
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
	Collapse repeated consecutive labels.
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



def _dominant_sequence_for_motif(
	labels,
	runs,
	motif_state,
	max_length=3,
):
	'''
	Estimate a representative ordered behavior sequence inside a motif.

	For every occurrence, consecutive duplicate behavior labels are
	collapsed. Common subsequences of length up to max_length are counted.
	'''

	counter=Counter()

	for state,start,end in runs:

		if state!=motif_state:
			continue

		collapsed=_collapse_repeats(
			labels[
				start:end
			]
		)

		if len(collapsed)==0:
			continue

		length=min(
			max_length,
			len(collapsed),
		)

		# Prefer longer behavioral sequences when available.
		while length>=1:

			if len(collapsed)>=length:

				for index in range(
					len(collapsed)
					-length
					+1
				):

					key=tuple(
						collapsed[
							index:index+length
						]
					)

					counter[
						(
							length,
							key,
						)
					]+=1

			length-=1

	if len(counter)==0:
		return ''

	maximum_length=max(
		key[0]
		for key in counter
	)

	candidates={
		key:count
		for key,count in counter.items()
		if key[0]==maximum_length
	}

	best=max(
		candidates,
		key=candidates.get,
	)

	return ' -> '.join(
		best[1]
	)



def _representative_intervals(
	runs,
	motif_state,
	time_points,
	limit=3,
):
	'''
	Return representative longest motif occurrences.
	'''

	candidates=[]

	for state,start,end in runs:

		if state!=motif_state:
			continue

		start_time=float(
			time_points[
				start
			]
		)

		end_index=min(
			end-1,
			len(time_points)-1,
		)

		end_time=float(
			time_points[
				end_index
			]
		)

		duration=max(
			0.0,
			end_time-start_time,
		)

		candidates.append(
			(
				duration,
				start_time,
				end_time,
			)
		)

	candidates.sort(
		key=lambda value:value[0],
		reverse=True,
	)

	intervals=[]

	for _,start_time,end_time in candidates[
		:limit
	]:

		intervals.append(
			f'{start_time:.3f}-{end_time:.3f}s'
		)

	return '; '.join(
		intervals
	)



def _motif_statistics(
	states,
	labels,
	time_points,
	composition,
	transition_matrix,
	behavior_names,
	top_n_behaviors,
):
	'''
	Generate motif-level summary statistics.
	'''

	states=np.asarray(
		states,
		dtype=int,
	)

	time_points=np.asarray(
		time_points,
		dtype=float,
	)

	n_motifs=composition.shape[0]

	runs=_contiguous_runs(
		states
	)

	if len(time_points)>1:

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

	for motif_state in range(
		n_motifs
	):

		motif_id=(
			motif_state+1
		)

		motif_runs=[
			(
				start,
				end,
			)
			for state,start,end in runs
			if state==motif_state
		]

		occurrences=len(
			motif_runs
		)

		frame_count=int(
			np.sum(
				states==motif_state
			)
		)

		occupancy=(
			frame_count
			/len(states)
			if len(states)>0
			else 0.0
		)

		durations=[
			(
				end-start
			)
			*frame_duration
			for start,end in motif_runs
		]

		mean_duration=(
			float(
				np.mean(
					durations
				)
			)
			if len(durations)>0
			else 0.0
		)

		probabilities=composition[
			motif_state
		]

		order=np.argsort(
			probabilities
		)[::-1]

		top_n=min(
			int(
				top_n_behaviors
			),
			len(
				behavior_names
			),
		)

		representative=[]

		for behavior_index in order[
			:top_n
		]:

			representative.append(
				str(
					behavior_names[
						behavior_index
					]
				)
				+' ('
				+format(
					float(
						probabilities[
							behavior_index
						]
					),
					'.3f',
				)
				+')'
			)

		representative_behaviors='; '.join(
			representative
		)

		dominant_sequence=(
			_dominant_sequence_for_motif(
				labels,
				runs,
				motif_state,
				max_length=3,
			)
		)

		preceding=[]

		following=[]

		for index,run in enumerate(
			runs
		):

			state,_,_=run

			if state!=motif_state:
				continue

			if index>0:

				preceding.append(
					runs[
						index-1
					][0]
					+1
				)

			if index<len(runs)-1:

				following.append(
					runs[
						index+1
					][0]
					+1
				)

		common_preceding=''

		if len(preceding)>0:

			common_preceding=int(
				Counter(
					preceding
				).most_common(
					1
				)[0][0]
			)

		common_following=''

		if len(following)>0:

			common_following=int(
				Counter(
					following
				).most_common(
					1
				)[0][0]
			)

		representative_intervals=(
			_representative_intervals(
				runs,
				motif_state,
				time_points,
				limit=3,
			)
		)

		# Excluding self-transition makes the most likely destination
		# easier to interpret.
		transition_row=transition_matrix[
			motif_state
		].copy()

		if len(transition_row)>1:

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

		rows.append(
			{
				'motif_id':motif_id,
				'motif_name':'Motif '+str(motif_id),
				'dominant_behavior_sequence':dominant_sequence,
				'representative_behaviors':representative_behaviors,
				'occupancy':float(occupancy),
				'mean_duration':float(mean_duration),
				'occurrences':int(occurrences),
				'frame_count':int(frame_count),
				'common_preceding_motif':common_preceding,
				'common_following_motif':common_following,
				'most_likely_next_motif':most_likely_next,
				'representative_intervals':representative_intervals,
			}
		)

	return pd.DataFrame(
		rows
	)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def _plot_timeline(
	time_points,
	states,
	output_path,
	animal_id,
	motif_names=None,
):
	'''
	Plot inferred motif state across the analyzed time sequence.
	'''

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
		figsize=(14,3.5)
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
			'Motif '+str(
				motif_id
			)
			for motif_id in unique_ids
		]

	else:

		labels=[
			motif_names.get(
				motif_id,
				'Motif '+str(
					motif_id
				),
			)
			for motif_id in unique_ids
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
		+str(animal_id)
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
	'''
	Plot motif x behavior composition heatmap.
	'''

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
			'Motif '+str(
				motif_id
			)
			for motif_id in range(
				1,
				n_motifs+1,
			)
		]

	else:

		y_labels=[
			motif_names.get(
				motif_id,
				'Motif '+str(
					motif_id
				),
			)
			for motif_id in range(
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
		+str(animal_id)
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


# ---------------------------------------------------------------------
# Main motif discovery
# ---------------------------------------------------------------------


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
	'''
	Discover recurring behavioral motifs.

	Each animal is analyzed independently.

	Workflow
	--------
	1. Prepare ordered frame-level behavior sequence.
	2. Build sliding-window behavior compositions.
	3. Fit Gaussian HMM.
	4. Decode one latent motif for every window.
	5. Map motif states back to every frame.
	6. Calculate motif statistics.
	7. Export summary tables and visualizations.

	Parameters
	----------
	event_probability : dict, list, tuple, or numpy.ndarray
		LabGym frame-level behavior sequence(s).

	time_points : sequence
		Frame timestamps.

	behavior_names : sequence
		Stable ordered list of behavior classes.

	output_folder : str
		Output directory.

	n_motifs : int
		Requested number of latent behavioral motifs.

	random_seed : int
		Random seed for reproducibility.

	max_iter : int
		Maximum number of Baum-Welch iterations.

	tol : float
		Convergence tolerance.

	top_n_behaviors : int
		Number of representative behaviors reported for each motif.

	window_size : int
		Number of frames in each temporal behavior-composition window.

	window_step : int
		Number of frames between consecutive windows.

	Returns
	-------
	dict
		Dictionary indexed by animal ID.
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

	if len(time_points)<2:

		raise ValueError(
			'At least two analyzed frames are required.'
		)

	behavior_names=[
		str(name)
		for name in behavior_names
	]

	if len(behavior_names)<2:

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

	if window_size>len(time_points):

		raise ValueError(
			'Window size cannot exceed the number of analyzed frames.'
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

		if len(events)!=len(
			time_points
		):

			raise ValueError(
				'ID '
				+str(animal_id)
				+': behavior sequence length ('
				+str(len(events))
				+') does not match time-point length ('
				+str(len(time_points))
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

		if len(window_features)<n_motifs:

			raise ValueError(
				'ID '
				+str(animal_id)
				+': requested '
				+str(n_motifs)
				+' motifs, but only '
				+str(len(window_features))
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

		composition=(
			_motif_composition(
				window_features,
				window_states,
				n_motifs,
				len(
					behavior_names
				),
			)
		)

		summary=(
			_motif_statistics(
				frame_states,
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
		)

		prefix=(
			'behavioral_motifs_ID'
			+str(animal_id)
		)

		# -----------------------------------------------------
		# Frame assignments
		# -----------------------------------------------------

		frame_table=pd.DataFrame(
			{
				'frame':np.arange(
					len(
						frame_states
					),
					dtype=int,
				),
				'time_seconds':time_points,
				'observed_behavior':labels,
				'motif_id':frame_states+1,
				'motif_name':[
					'Motif '
					+str(
						state+1
					)
					for state in frame_states
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

		# -----------------------------------------------------
		# Window assignments
		# -----------------------------------------------------

		window_table=pd.DataFrame(
			window_features,
			columns=behavior_names,
		)

		window_table.insert(
			0,
			'window_end_frame',
			[
				end-1
				for _,end in window_bounds
			],
		)

		window_table.insert(
			0,
			'window_start_frame',
			[
				start
				for start,_ in window_bounds
			],
		)

		window_table.insert(
			0,
			'window_center_frame',
			window_centers,
		)

		window_table[
			'motif_id'
		]=window_states+1

		window_table[
			'motif_name'
		]=[
			'Motif '
			+str(
				state+1
			)
			for state in window_states
		]

		window_table.to_csv(
			output_folder
			/(
				prefix
				+'_window_assignments.csv'
			),
			index=False,
		)

		# -----------------------------------------------------
		# Summary
		# -----------------------------------------------------

		summary.to_csv(
			output_folder
			/(
				prefix
				+'_summary.csv'
			),
			index=False,
		)

		# -----------------------------------------------------
		# Composition / emission table
		# -----------------------------------------------------

		emission_table=pd.DataFrame(
			composition,
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
				for motif_id in range(
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

		# -----------------------------------------------------
		# Transition probabilities
		# -----------------------------------------------------

		transition_labels=[
			'Motif '
			+str(
				motif_id
			)
			for motif_id in range(
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

		# -----------------------------------------------------
		# Visualizations
		# -----------------------------------------------------

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
			composition,
			behavior_names,
			output_folder
			/(
				prefix
				+'_composition_heatmap.png'
			),
			animal_id,
		)

		# -----------------------------------------------------
		# Model data
		# -----------------------------------------------------

		np.savez_compressed(
			output_folder
			/(
				prefix
				+'_model.npz'
			),
			start_probability=model.startprob_,
			transition_probability=model.transmat_,
			window_means=model.means_,
			window_covariances=model.covars_,
			window_features=window_features,
			window_centers=window_centers,
			decoded_window_motif=window_states+1,
			decoded_frame_motif=frame_states+1,
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

		# -----------------------------------------------------
		# Metadata
		# -----------------------------------------------------

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
			'animal_id':metadata_animal_id,
			'model_type':'window_composition_gaussian_hmm',
			'n_motifs':n_motifs,
			'window_size_frames':window_size,
			'window_step_frames':window_step,
			'random_seed':random_seed,
			'maximum_iterations':max_iter,
			'iterations':len(
				model.log_likelihood_history_
			),
			'final_log_likelihood':(
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
			'n_frames':int(
				len(
					frame_states
				)
			),
			'n_windows':int(
				len(
					window_features
				)
			),
			'missing_frames_imputed':int(
				missing_count
			),
			'behavior_names':behavior_names,
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
			'model':model,
			'states':frame_states,
			'window_states':window_states,
			'window_features':window_features,
			'window_centers':window_centers,
			'window_bounds':window_bounds,
			'composition':composition,
			'summary':summary,
			'metadata':metadata,
		}

	return all_results


# ---------------------------------------------------------------------
# User-defined motif names
# ---------------------------------------------------------------------


def apply_motif_names(
	output_folder,
	animal_id,
	motif_names,
):
	'''
	Apply user-defined names to previously inferred behavioral motifs.

	The naming operation changes interpretation labels only. It does not
	refit or otherwise alter the Hidden Markov Model.

	motif_names example:

	{
		1: 'Exploration',
		2: 'Feeding',
		3: 'Grooming',
		4: 'Resting',
	}
	'''

	output_folder=Path(
		output_folder
	)

	prefix=(
		'behavioral_motifs_ID'
		+str(animal_id)
	)

	clean_names={}

	for motif_id,motif_name in motif_names.items():

		motif_id=int(
			motif_id
		)

		motif_name=str(
			motif_name
		).strip()

		if len(motif_name)==0:

			raise ValueError(
				'Motif '
				+str(motif_id)
				+' must have a name.'
			)

		clean_names[
			motif_id
		]=motif_name

	if len(
		set(
			name.lower()
			for name in clean_names.values()
		)
	)!=len(
		clean_names
	):

		raise ValueError(
			'Every motif must have a unique name.'
		)

	# ---------------------------------------------------------
	# Summary
	# ---------------------------------------------------------

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
		.astype(int)
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
			.astype(int)
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

	if 'common_preceding_motif' in summary.columns:

		def preceding_name(value):

			if pd.isna(value):
				return ''

			try:

				return clean_names.get(
					int(value),
					'',
				)

			except (
				TypeError,
				ValueError,
			):

				return ''

		summary[
			'common_preceding_motif_name'
		]=summary[
			'common_preceding_motif'
		].apply(
			preceding_name
		)

	if 'common_following_motif' in summary.columns:

		def following_name(value):

			if pd.isna(value):
				return ''

			try:

				return clean_names.get(
					int(value),
					'',
				)

			except (
				TypeError,
				ValueError,
			):

				return ''

		summary[
			'common_following_motif_name'
		]=summary[
			'common_following_motif'
		].apply(
			following_name
		)

	summary.to_csv(
		summary_path,
		index=False,
	)

	# ---------------------------------------------------------
	# Frame assignments
	# ---------------------------------------------------------

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

		if 'motif_id' in frame_table.columns:

			frame_table[
				'motif_name'
			]=(
				frame_table[
					'motif_id'
				]
				.astype(int)
				.map(
					clean_names
				)
			)

			frame_table.to_csv(
				frame_path,
				index=False,
			)

	# ---------------------------------------------------------
	# Window assignments
	# ---------------------------------------------------------

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

		if 'motif_id' in window_table.columns:

			window_table[
				'motif_name'
			]=(
				window_table[
					'motif_id'
				]
				.astype(int)
				.map(
					clean_names
				)
			)

			window_table.to_csv(
				window_path,
				index=False,
			)

	# ---------------------------------------------------------
	# Emission/composition table
	#
	# Written to be idempotent. Running motif naming repeatedly will
	# update columns instead of trying to insert duplicate columns.
	# ---------------------------------------------------------

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
			for column in emission.columns
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

		expected_ids=np.arange(
			1,
			len(
				emission
			)+1,
			dtype=int,
		)

		if 'motif_id' in emission.columns:

			emission[
				'motif_id'
			]=expected_ids

		else:

			emission.insert(
				0,
				'motif_id',
				expected_ids,
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
			for motif_id in expected_ids
		]

		if 'motif_name' in emission.columns:

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
			for column in emission.columns
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

	# ---------------------------------------------------------
	# Transition matrix
	# ---------------------------------------------------------

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

		if transition.shape[0]!=transition.shape[1]:

			raise ValueError(
				'Motif transition probability table must be square; '
				'found shape '
				+str(
					transition.shape
				)
				+'.'
			)

		ordered_names=[
			clean_names.get(
				motif_id,
				'Motif '
				+str(
					motif_id
				),
			)
			for motif_id in range(
				1,
				len(
					transition
				)+1,
			)
		]

		transition.index=ordered_names
		transition.columns=ordered_names

		transition.to_csv(
			transition_path,
			index=True,
		)

	# ---------------------------------------------------------
	# Metadata
	# ---------------------------------------------------------

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
		):motif_name
		for motif_id,motif_name in clean_names.items()
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
	# Regenerate timeline
	# ---------------------------------------------------------

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

			if 'time_seconds' in frame_table.columns:

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
				.astype(int)
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
				motif_names=clean_names,
			)

	# ---------------------------------------------------------
	# Regenerate composition heatmap
	# ---------------------------------------------------------

	if emission_path.exists():

		emission=pd.read_csv(
			emission_path
		)

		behavior_columns=[
			column
			for column in emission.columns
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
				motif_names=clean_names,
			)

	return clean_names

def _fill_missing_behaviors(events):
	'''
	Convert LabGym event entries into a clean behavior-label sequence.

	LabGym parse_all_events_file() returns entries such as:

		['sniffing sv', 0.92]
		['NA', -1]

	This function extracts the behavior name, treats NA as missing,
	then forward/backward fills missing frames.

	Returns
	-------
	cleaned : list[str]
		Clean frame-level behavior labels.

	missing_count : int
		Number of frames that originally contained NA/missing behavior.
	'''

	events=list(events)

	cleaned=[]

	for event in events:

		# -----------------------------------------------------
		# LabGym standard format:
		# [behavior_name, probability]
		# -----------------------------------------------------

		if isinstance(
			event,
			(
				list,
				tuple,
				np.ndarray,
			),
		):

			if len(event)==0:

				cleaned.append(None)
				continue

			behavior=event[0]

		else:

			# Also support direct string labels.
			behavior=event

		# -----------------------------------------------------
		# Missing values
		# -----------------------------------------------------

		if behavior is None:

			cleaned.append(None)
			continue

		if isinstance(
			behavior,
			(
				float,
				np.floating,
			),
		):

			if np.isnan(behavior):

				cleaned.append(None)
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

			cleaned.append(None)

		else:

			cleaned.append(
				behavior
			)

	missing_count=sum(
		value is None
		for value in cleaned
	)

	if missing_count==len(cleaned):

		raise ValueError(
			'The behavior sequence contains no valid behavior labels.'
		)

	# ---------------------------------------------------------
	# Forward-fill missing frames
	# ---------------------------------------------------------

	last_valid=None

	for index,value in enumerate(
		cleaned
	):

		if value is None:

			if last_valid is not None:

				cleaned[index]=last_valid

		else:

			last_valid=value

	# ---------------------------------------------------------
	# Backward-fill missing frames at beginning
	# ---------------------------------------------------------

	next_valid=None

	for index in range(
		len(cleaned)-1,
		-1,
		-1,
	):

		if cleaned[index] is None:

			if next_valid is not None:

				cleaned[index]=next_valid

		else:

			next_valid=cleaned[index]

	return cleaned,missing_count