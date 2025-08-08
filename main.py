import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
import os

# Step 1: Load MIDI files and extract notes
def get_notes(midi_folder):
    notes = []
    for file in glob.glob(f"{midi_folder}/*.mid"):
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)

        if parts:  # File has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:  # File has no instrument parts (single stream)
            notes_to_parse = midi.recurse()

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

# Step 2: Prepare sequences
def prepare_sequences(notes, sequence_length):
    pitchnames = sorted(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[note] for note in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(len(pitchnames))
    network_output = to_categorical(network_output)
    
    return network_input, network_output, note_to_int, pitchnames

# Step 3: Build model
def build_model(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Step 4: Generate music
def generate_music(model, network_input, note_to_int, pitchnames, output_path):
    int_to_note = dict((number, note) for note, number in note_to_int.items())
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    pattern = pattern.tolist()
    prediction_output = []

    for note_index in range(200):
        input_seq = np.reshape(pattern, (1, len(pattern), 1))
        input_seq = input_seq / float(len(pitchnames))

        prediction = model.predict(input_seq, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append([index])
        pattern = pattern[1:]

    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            chord_notes = [note.Note(int(n)) for n in notes_in_chord]
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=os.path.join(output_path, 'generated.mid'))
    print(f" Music generated and saved to {output_path}/generated.mid")

# Main runner
def main():
    print(" Loading MIDI files...")
    notes = get_notes("midi_songs")

    if len(notes) < 100:
        print(" Not enough notes to train a model. Add more MIDI files.")
        return

    print(f" Notes extracted: {len(notes)}")
    sequence_length = 50
    X, y, note_to_int, pitchnames = prepare_sequences(notes, sequence_length)

    print(" Building and training model...")
    model = build_model(X, len(pitchnames))
    model.fit(X, y, epochs=20, batch_size=64)

    if not os.path.exists("output"):
        os.makedirs("output")

    print(" Generating music...")
    generate_music(model, X, note_to_int, pitchnames, "output")

if __name__ == "__main__":
    main()
    