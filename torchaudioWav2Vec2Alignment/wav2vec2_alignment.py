# wav2vec2_alignment.py
# Tutorial on how to align text transcript to speech with torchaudio
# and using the CTC segmentation algorithm described in this paper:
# CTC-Segmentation of Large Corpora for German End-to-end Speech 
# Recognition. 
# Source: https://pytorch.org/audio/stable/tutorials/forced_alignment_
#   tutorial.html
# Source: https://arxiv.org/pdf/2007.09127.pdf
# Pytorch 2.0
# Windows/MacOS/Linux
# Python 3.7


from dataclasses import dataclass
import os
import shutil
import IPython

import torch
import torchaudio
import matplotlib
import matplotlib.pyplot as plt

# Set figure size in runtime configuration parameters.
matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

# Set seed.
torch.random.manual_seed(0)


def main():
	# Print pytorch versions (should work for 2.0+).
	print(torch.__version__)
	print(torchaudio.__version__)

	# Set device.
	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		torch_device = "cuda"
	elif torch.backends.mps.is_available():
		torch_device = "mps"
	else:
		torch_device = "cpu"
	device = torch.device(torch_device)
	print(device)

	# Download/load sample audio file.
	saved_speech_file = './Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav'
	if os.path.exists(saved_speech_file):
		print("Loading sample audio file from local storage.")
		SPEECH_FILE = saved_speech_file
	else:
		print("Downloading sample audio file.")
		SPEECH_FILE = torchaudio.utils.download_asset(
			"tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
		)

		print("Copying downloaded audio file to local storage.")
		shutil.copy(SPEECH_FILE, saved_speech_file)
	print(f"Sample audio file: {SPEECH_FILE}")

	# Overview
	# The process of alignment looks like the following:
	#	1. Estimate the frame-wise label probability from audio 
	#		waveform.
	#	2. Generate the trellis matrix which represents the probability
	#		of labels aligned at time step.
	#	3. Find the most likely path from the trellis matrix.
	# In this example, we use torchaudio’s Wav2Vec2 model for acoustic 
	# feature extraction.

	# Generate frame-wise label probability.
	# The first step is to generate the label class probability of each
	# audio frame. We can use a Wav2Vec2 model that is trained for ASR.
	# Here we use torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H(). 
	# torchaudio provides easy access to pretrained models with 
	# associated labels.
	bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
	model = bundle.get_model().to(device)
	labels = bundle.get_labels()
	with torch.inference_mode():
		waveform, _ = torchaudio.load(SPEECH_FILE)
		emissions, _ = model(waveform.to(device))

		# In the subsequent sections, we will compute the probability 
		# in log-domain to avoid numerical instability. For this 
		# purpose, we normalize the emission with torch.log_softmax().
		emissions = torch.log_softmax(emissions, dim=-1)

	emission = emissions[0].cpu().detach()

	# Visualization
	print(f"labels: {labels}")
	plt.imshow(emission.T)
	plt.colorbar()
	plt.title("Frame-wise class probability")
	plt.xlabel("Time")
	plt.ylabel("Labels")
	# plt.show()
	plt.savefig('frame-wise_class_probability.png')

	# Generate alignment probability (trellis)
	# From the emission matrix, next we generate the trellis which 
	# represents the probability of transcript labels occur at each 
	# time frame.
	# Trellis is 2D matrix with time axis and label axis. The label 
	# axis represents the transcript that we are aligning. In the 
	# following, we use t to denote the index in time axis and j to 
	# denote the index in label axis. c j represents the label at label
	# index j.
	# To generate, the probability of time step t+1, we look at the 
	# trellis from time step t and emission at time step t+1. There are
	# two path to reach to time step t+1 with label c j+1. The first 
	# one is the case where the label was c j+1 at t and there was no 
	# label change from t to t+1. The other case is where the label was
	# c j at t and it transitioned to the next label c j+1 at t+1.
	# Since we are looking for the most likely transitions, we take the
	# more likely path for the value of k (t+1,j+1), that is
	# k (t+1,j+1) =max(k (t,j) p(t+1,c j+1),k (t,j+1) p(t+1,repeat))
	# where k represents is trellis matrix, and p(t,c j) represents the
	# probability of label c j at time step t. repeat represents the 
	# blank token from CTC formulation. 
	transcript = "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT"
	dictionary = {c: i for i, c in enumerate(labels)}

	tokens = [dictionary[c] for c in transcript]
	print(list(zip(transcript, tokens)))


	def get_trellis(emission, tokens, blank_id=0):
		num_frame = emission.size(0)
		num_tokens = len(tokens)

		# Trellis has extra diemsions for both time axis and tokens.
		# The extra dim for tokens represents <SoS> (start-of-sentence)
		# The extra dim for time axis is for simplification of the code.
		trellis = torch.empty((num_frame + 1, num_tokens + 1))
		trellis[0, 0] = 0
		trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
		trellis[0, -num_tokens:] = -float("inf")
		trellis[-num_tokens:, 0] = float("inf")

		for t in range(num_frame):
			trellis[t + 1, 1:] = torch.maximum(
				# Score for staying at the same token
				trellis[t, 1:] + emission[t, blank_id],
				# Score for changing to the next token
				trellis[t, :-1] + emission[t, tokens],
			)
		return trellis
	

	trellis = get_trellis(emission, tokens)

	# Visualization
	plt.imshow(trellis[1:, 1:].T, origin="lower")
	plt.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
	plt.colorbar()
	# plt.show()
	plt.savefig("trellis.png")

	# In the above visualization, we can see that there is a trace of 
	# high probability crossing the matrix diagonally.

	# Find the most likely path (backtracking)
	# Once the trellis is generated, we will traverse it following the 
	# elements with high probability.
	# We will start from the last label index with the time step of 
	# highest probability, then, we traverse back in time, picking 
	# stay (c j → c j) or transition (c j → c j+1), based on the 
	# post-transition probability k t,j p(t+1,c j+1) or 
	# k t,j+1 p(t+1,repeat).
	# Transition is done once the label reaches the beginning.
	# The trellis matrix is used for path-finding, but for the final 
	# probability of each segment, we take the frame-wise probability 
	# from emission matrix.


	@dataclass
	class Point:
		token_index: int
		time_index: int
		score: float


	def backtrack(trellis, emission, tokens, blank_id=0):
		# Note:
		# j and t are indices for trellis, which has extra dimensions
		# for time and tokens at the beginning.
		# When referring to time frame index `T` in trellis,
		# the corresponding index in emission is `T-1`.
		# Similarly, when referring to token index `J` in trellis,
		# the corresponding index in transcript is `J-1`.
		j = trellis.size(1) - 1
		t_start = torch.argmax(trellis[:, j]).item()

		path = []
		for t in range(t_start, 0, -1):
			# 1. Figure out if the current position was stay or change
			# Note (again):
			# `emission[J-1]` is the emission at time frame `J` of trellis dimension.
			# Score for token staying the same from time frame J-1 to T.
			stayed = trellis[t - 1, j] + emission[t - 1, blank_id]

			# Score for token changing from C-1 at T-1 to J at T.
			changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

			# 2. Store the path with frame-wise probability.
			prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()

			# Return token index and time index in non-trellis coordinate.
			path.append(Point(j - 1, t - 1, prob))

			# 3. Update the token
			if changed > stayed:
				j -= 1
				if j == 0:
					break
		else:
			raise ValueError("Failed to align")
		return path[::-1]
	

	path = backtrack(trellis, emission, tokens)
	for p in path:
		print(p)


	# Visualization
	def plot_trellis_with_path(trellis, path):
		# To plot trellis with path, we take advantage of 'nan' value
		trellis_with_path = trellis.clone()
		for _, p in enumerate(path):
			trellis_with_path[p.time_index, p.token_index] = float("nan")
		plt.imshow(trellis_with_path[1:, 1:].T, origin="lower")


	plot_trellis_with_path(trellis, path)
	plt.title("The path found by backtracking")
	# plt.show()
	plt.savefig("backtracking.png")

	# Looking good. Now this path contains repetations for the same 
	# labels, so let’s merge them to make it close to the original 
	# transcript.
	# When merging the multiple path points, we simply take the average
	# probability for the merged segments.


	# Merge the labels
	@dataclass
	class Segment:
		label: str
		start: int
		end: int
		score: float

		def __repr__(self):
			return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

		@property
		def length(self):
			return self.end - self.start
		

	def merge_repeats(path):
		i1, i2 = 0, 0
		segments = []
		while i1 < len(path):
			while i2 < len(path) and path[i1].token_index == path[i2].token_index:
				i2 += 1
			score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
			segments.append(
				Segment(
					transcript[path[i1].token_index],
					path[i1].time_index,
					path[i2 - 1].time_index + 1,
					score,
				)
			)
			i1 = i2
		return segments


	segments = merge_repeats(path)
	for seg in segments:
		print(seg)


	# Visualization
	def plot_trellis_with_segments(trellis, segments, transcript):
		# To plot trellis with path, we take advantage of 'nan' value
		trellis_with_path = trellis.clone()
		for i, seg in enumerate(segments):
			if seg.label != "|":
				trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

		fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))
		ax1.set_title("Path, label and probability for each label")
		ax1.imshow(trellis_with_path.T, origin="lower")
		ax1.set_xticks([])

		for i, seg in enumerate(segments):
			if seg.label != "|":
				ax1.annotate(seg.label, (seg.start + 0.7, i + 0.3), weight="bold")
				ax1.annotate(f"{seg.score:.2f}", (seg.start - 0.3, i + 4.3))

		ax2.set_title("Label probability with and without repetation")
		xs, hs, ws = [], [], []
		for seg in segments:
			if seg.label != "|":
				xs.append((seg.end + seg.start) / 2 + 0.4)
				hs.append(seg.score)
				ws.append(seg.end - seg.start)
				ax2.annotate(seg.label, (seg.start + 0.8, -0.07), weight="bold")
		ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

		xs, hs = [], []
		for p in path:
			label = transcript[p.token_index]
			if label != "|":
				xs.append(p.time_index + 1)
				hs.append(p.score)

		ax2.bar(xs, hs, width=0.5, alpha=0.5)
		ax2.axhline(0, color="black")
		ax2.set_xlim(ax1.get_xlim())
		ax2.set_ylim(-0.1, 1.1)


	plot_trellis_with_segments(trellis, segments, transcript)
	plt.tight_layout()
	# plt.show()
	plt.savefig("trellis_with_segments.png")

	# Looks good. Now let’s merge the words. The Wav2Vec2 model uses 
	# '|' as the word boundary, so we merge the segments before each 
	# occurance of '|'.
	# Then, finally, we segment the original audio into segmented audio
	# and listen to them to see if the segmentation is correct.


	# Merge words
	def merge_words(segments, separator="|"):
		words = []
		i1, i2 = 0, 0
		while i1 < len(segments):
			if i2 >= len(segments) or segments[i2].label == separator:
				if i1 != i2:
					segs = segments[i1:i2]
					word = "".join([seg.label for seg in segs])
					score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
					words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
				i1 = i2 + 1
				i2 = i1
			else:
				i2 += 1
		return words


	word_segments = merge_words(segments)
	for word in word_segments:
		print(word)


	# Visualization
	def plot_alignments(trellis, segments, word_segments, waveform):
		trellis_with_path = trellis.clone()
		for i, seg in enumerate(segments):
			if seg.label != "|":
				trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

		fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))

		ax1.imshow(trellis_with_path[1:, 1:].T, origin="lower")
		ax1.set_xticks([])
		ax1.set_yticks([])

		for word in word_segments:
			ax1.axvline(word.start - 0.5)
			ax1.axvline(word.end - 0.5)

		for i, seg in enumerate(segments):
			if seg.label != "|":
				ax1.annotate(seg.label, (seg.start, i + 0.3))
				ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 4), fontsize=8)

		# The original waveform
		ratio = waveform.size(0) / (trellis.size(0) - 1)
		ax2.plot(waveform)
		for word in word_segments:
			x0 = ratio * word.start
			x1 = ratio * word.end
			ax2.axvspan(x0, x1, alpha=0.1, color="red")
			ax2.annotate(f"{word.score:.2f}", (x0, 0.8))

		for seg in segments:
			if seg.label != "|":
				ax2.annotate(seg.label, (seg.start * ratio, 0.9))
		xticks = ax2.get_xticks()
		plt.xticks(xticks, xticks / bundle.sample_rate)
		ax2.set_xlabel("time [second]")
		ax2.set_yticks([])
		ax2.set_ylim(-1.0, 1.0)
		ax2.set_xlim(0, waveform.size(-1))


	plot_alignments(
		trellis,
		segments,
		word_segments,
		waveform[0],
	)
	# plt.show()
	plt.savefig("trellis_alignments.png")


	# A trick to embed the resulting audio to the generated file.
	# `IPython.display.Audio` has to be the last call in a cell,
	# and there should be only one call par cell.
	def display_segment(i):
		ratio = waveform.size(1) / (trellis.size(0) - 1)
		word = word_segments[i]
		x0 = int(ratio * word.start)
		x1 = int(ratio * word.end)
		print(f"{word.label} ({word.score:.2f}): {x0 / bundle.sample_rate:.3f} - {x1 / bundle.sample_rate:.3f} sec")
		segment = waveform[:, x0:x1]
		# return IPython.display.Audio(segment.numpy(), rate=bundle.sample_rate)
	

	# Generate the audio for each segment
	print(transcript)
	# IPython.display.Audio(SPEECH_FILE)

	for i in range(len(transcript.split('|'))):
		display_segment(i)

	# Conclusion
	# In this tutorial, we looked how to use torchaudio’s Wav2Vec2 
	# model to perform CTC segmentation for forced alignment.
	# Total running time of the script: ( 0 minutes 3.182 seconds)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()