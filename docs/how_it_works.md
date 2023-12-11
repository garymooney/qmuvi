The quantum circuits are run on [Qiskit Aer’s simulator](https://qiskit.org/documentation/tutorials/simulators/1_aer_provider.html) which supports noise models ([NoiseModel](https://qiskit.org/documentation/stubs/qiskit.providers.aer.noise.NoiseModel.html)) that are used to mimic noise in a physical quantum device. For your quantum circuit, you are able to specify a noise model to be applied to the quantum simulation during qMuVi's sampling of the states. This is particularly useful in understanding how the noise present on a real quantum computer will affect the outcome of your states.

Various instruments that play your music can be selected easily using the get_instruments() method, with a range of predefined collections, including piano, tuned percussion, organ, guitar etc. Behind the scenes, the instruments are assigned as integers according to the [General MIDI](https://en.wikipedia.org/wiki/General_MIDI) standard, which the [Mido Python library](https://mido.readthedocs.io/en/latest/index.html) uses to generate the music using the specified digital instruments.

There are three note maps that are provided by default, the chromatic C scale, the C major scale and the F minor scale, which are used to map the quantum basis state numbers (e.g. |0>, |2>, |7>, etc...) to MIDI note numbers. For example, the C major scale map works by adding 60 to the state number so that the |0> basis state is mapped to middle C, then rounds the note number down to the nearest note in the C major scale. This mapping is readily customised by defining your own method that maps an _int_ to an _int_.

Another important part to music is the rhythm. The note play durations, as well as the rest times after the notes, are defined as a list of tuples. The tuples specify the play and rest time in units of ticks (where 480 ticks is 1 second) for the sound samples of the quantum state.

The [MoviePy Python library](https://zulko.github.io/moviepy/) is used to render the music videos which display a visual representation of your input circuit. The quantum state is visualised by animating figures that show various important information, such as the probability distribution of basis states for each pure state, with colours representing their phases.

Once your quantum circuit, instruments and rhythm are defined (and optionally noise model and note map), you can input these parameters into methods such as  _make_music_video()_ or _make_music_midi()_ to generate a music video file or a raw MIDI file respectively. See below for code examples.