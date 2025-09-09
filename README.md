# AI Music Generator 🎵

A web-based music generation app built with Streamlit and TensorFlow. This project uses an LSTM neural network trained on classical MIDI files to generate new musical sequences. Users can generate, view, and download new MIDI music with a single click.

## Features
- Generate new music using a trained LSTM model
- Download generated music as a MIDI file
- View generated notes as text and in a table
- Simple, interactive web interface (Streamlit)

## How it works
1. The app loads a pre-trained LSTM model and note mappings.
2. When you click "Generate Music," it creates a new sequence of notes.
3. The generated music is saved as a MIDI file, which you can download and play.

## Requirements
- Python 3.8+
- TensorFlow
- music21
- Streamlit
- numpy
- pandas

## Getting Started
1. Clone this repository.
2. Place your trained model (`s2s.keras`), note mapping (`ind2note.pkl`), and test data (`x_test.npy`) in the project folder.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```

## License
MIT License

---
Feel free to customize this README for your project!
