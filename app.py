import streamlit as st
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
#!pip install tensorflow
st.title("Music Generation App 🎵")

# Check for required files
required_files = ["s2s.keras", "ind2note.pkl", "x_test.npy"]
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
	st.error(f"Missing files: {', '.join(missing)}. Please ensure all files are in the app directory.")
	st.stop()

try:
	model = load_model("s2s.keras")
	with open("ind2note.pkl", "rb") as f:
		ind2note = pickle.load(f)
	x_test = np.load("x_test.npy")
except Exception as e:
	st.error(f"Error loading files: {e}")
	st.stop()

st.success("Model and data loaded successfully!")

if st.button("Generate Music"):

	# Music generation logic
	import io
	from music21 import note, chord, instrument, stream

	# Pick a random seed pattern from x_test
	index = np.random.randint(0, len(x_test) - 1)
	music_pattern = x_test[index]
	out_pred = []
	timesteps = music_pattern.shape[0]

	# Generate 200 notes
	for i in range(200):
		music_pattern_reshaped = music_pattern.reshape(1, timesteps, 1)
		pred_index = np.argmax(model.predict(music_pattern_reshaped, verbose=0))
		out_pred.append(ind2note[pred_index])
		music_pattern = np.append(music_pattern, pred_index)
		music_pattern = music_pattern[1:]

	# Convert predictions to music21 stream
	output_notes = []
	for offset, pattern in enumerate(out_pred):
		if ('.' in pattern) or pattern.isdigit():
			notes_in_chord = pattern.split('.')
			notes = []
			for current_note in notes_in_chord:
				i_curr_note = int(current_note)
				new_note = note.Note(i_curr_note)
				new_note.storedInstrument = instrument.Piano()
				notes.append(new_note)
			new_chord = chord.Chord(notes)
			new_chord.offset = offset
			output_notes.append(new_chord)
		else:
			new_note = note.Note(pattern)
			new_note.offset = offset
			new_note.storedInstrument = instrument.Piano()
			output_notes.append(new_note)

	midi_stream = stream.Stream(output_notes)
	midi_buffer = io.BytesIO()
	# Save to a temporary file, then read as bytes
	with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
		midi_stream.write('midi', fp=tmp.name)
		tmp.seek(0)
		midi_bytes = tmp.read()

	st.success("Music generated!")
	# Display generated notes as text
	st.subheader("Generated Notes (as text)")
	st.write(", ".join(out_pred))

	# Display generated notes as a table
	import pandas as pd
	df_notes = pd.DataFrame({"Note/Chord": out_pred})
	st.subheader("Generated Notes (as table)")
	st.dataframe(df_notes)

	st.download_button(
		label="Download MIDI file",
		data=midi_bytes,
		file_name="generated_music.mid",
		mime="audio/midi"
	)

st.write("Ready to generate music!")
