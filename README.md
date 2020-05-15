# Text Recogniton in News Video frames using EAST text detection and CRNN recognition
Text is everywhere as we know it, some places we encounter it mostly are for example books, newspapers,
invoices, hoardings/billboards, in movie subtitles,etc.

For many of these types of texts such as invoices the bills are black & white, have common fonts,
and have problems such as tilt in the picture, maybe 2-3 degree or maybe 90/180/270 degree flip,
have some kind of blurring, etc. These problems can be solved by detecting tilt by hough transform,
using gaussian blurring, sharpening, etc and then using the majorly important tool **tesseract**.

![invoice image](https://imgur.com/a/JTCZTXq)
But, for images in the wild like in stop signs billboard hoardings, for text-recognition, we have to 
resort to deep learning based approaches.

![Text in the wild - Photo by Marija Zaric on Unsplash](https://imgur.com/a/Iy8GaKb)

### Aim 

I aim to build a pipeline of a text-detection algorithm followed by text-recognition based on deep
learning and hope to accomplish the following tasks, also wish to compare to other paid solutions,
in the long term. 
The task list is below, please comment & give feedback as to any features I should add, and also
any critical feedback!

**Python version used - 3.6**

Task List:
- [x] Text Detection using EAST algorithm ( Got to use maybe a better algo) on all video frames by sampling.
- [x] Text recognition using CRNN on the segments detected from the frames & then printing text recognized frame by frame.
- [ ] Add text datagenerator script for people looking to do transfer learning on CRNN model.
- [ ] Try training part, if possible with Mysynth dataset (10 Gb (-_-)) and our custom data._
- [ ] Make a viable flask app.
- [ ] Add the app on docker and also on heroku to host it ( I guess ).
- [ ] Update to tensorflow 2.x versions.
- [ ] Create a tflite-app
- [ ] Aim to increase accuracy, since presently accuracy doesn't strike me as good enough :sweat:


### <font color='green'><----- Installation of Tensorflow 1.8.0 ------></font> 
Please follow the instructions at [Install older versions of tensorflow not available directly through pip](https://stackoverflow.com/a/41942396/8030107)
