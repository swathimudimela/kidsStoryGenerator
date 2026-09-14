Junior Directors: 
This app lets kids to direct their own movies. Kids just need to tell the story and see their imagination turn into a movie. 

Workflow : 
  1. Let the kids record their story. Kids narration has creative sound effects, pauses, and also their own words for certain things
  2. So use LLM with a little bit of creativity to transcribe the story.
  3. Once transcribed its passed to LLM to create a screenplay
  4. Then we use KOKORO-82M model to generate narration and image files that suits the transcript
  5. In the last step used moviepy to generate movie with the audio and image files.
