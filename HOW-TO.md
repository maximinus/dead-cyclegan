
Dead-Cyclegan Guide
===================

Basic Overview
--------------

The aim of the software is to train a network to turn audio from one domain - Grateful Dead audience tapes - into another - Grateful Dead soundboard tapes.

To perform this, we use a cyclegan.


Data Preparation
----------------

You will need to start with some audio in wav format. Almost certainly you will have complete shows, and they need to put in the `data/original` folder. The folder name must end in `AUD` or `SBD`, depending on what the source is.

As an example, I have a folder `./data/Original/1977-05-17-AUD` and it contains files such as `gd1977-05-17d1t001.wav`. it does not matter what the name of the file is in the sub-folder, it just needs to be a stereo wav file at 44,100 Hz.

It is strongly advised to remove all non-audio tracks. You should also trim any non-audio from the start and end of the track. The code will remove silence but audience tapes are never silent.


Running The Code
----------------

TODO


Generating A Trial Audio
------------------------

Once you have a model.
