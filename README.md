# 🎵 KaraFan  [![img](https://img.shields.io/github/stars/Captain-FLAM/KaraFan?color=ff0080&style=for-the-badge)](https://github.com/Captain-FLAM/KaraFan/stargazers) [![img](https://img.shields.io/github/license/Captain-FLAM/KaraFan?style=for-the-badge)](https://github.com/Captain-FLAM/KaraFan/blob/master/LICENSE)

The BEST music separation model (SDR++) ... to my ears ! 👂👂

As you've guessed, it was made specially for Karaoke (▶️ focus on musical part).

This project is open to all goodwill.

You can reach me by [email](https://github.com/Captain-FLAM) or Join all of us on this  [![Image](images/discord.svg) Discord server](https://discord.com/channels/708579735583588363/887455924845944873) !

# 📖 Table of Contents

- [🔥 INTRODUCTION](#-introduction)
- [💤 AN OLD DREAM](#-an-old-dream)
- [🧒 BIOGRAPHY](#-biography)
- [📆 HISTORY](#-history)
- [🚀 INSTALLATION](#-installation)
- [💡 TECHNICAL DETAILS](#-technical-details)
- [💗 SPECIAL THANKS TO...](#-special-thanks-to)
- [🦄 IN THE NEAR FUTURE](#-in-the-near-future)
- [📝 TODO](#-todo)
- [📜 LICENSE](#-license)

# 🔥 INTRODUCTION

I'm an amateur Rock singer who has often been disappointed by the inability to find songs of my favorite singers in the vast KAR databases I possess, or in the extensive library of thousands of songs offered by « KaraFun » (with a paid subscription).

Of course, you have all the singers's best-of, but it's not always the songs you want to sing.

And if the singer is not very famous, you'll find 3-4 songs with real musicians playing or you will have to sing on a MIDI file, which is not very pleasant (even with "Sound Fonts"), or you can forget it ...

# 💤 AN OLD DREAM

Since my childhood, I sing everytime and I've been dreaming of a software that would allow me to remove the voice of a song to sing on it.

TODAY, I'M 54 YEARS OLD, I'M STILL SINGING, AND MY DREAM HAS COME TRUE !

Now, I have the ability to create my own « **KFN** » files for my favorite songs, featuring real musicians who played on the original track, and use them with my beloved and renowned software, « KaraFun » 🎤💋.

Perhaps it's time to turn on your dreams too ... ?

# 🧒 BIOGRAPHY

Programmer since the age of 12 (1981).  
Before, I developed in ASM, C, C++, Basic, Visual Basic.  
Since the year 2000, I have been coding in PHP, MySQL, JavaScript, jQuery, HTML, CSS.  
And today in Python. ❤️

# 📆 HISTORY

I first started with the Demucs facebook research model, but I was disappointed with the results, especially with the instrumental part.

Then I discovered the MDX model, and I was amazed by the quality of the results, especially with the vocals and the instrumental part.

But I was still disappointed with the instrumental part, which was not shining enough for me.

So I decided to create my own process, based on the MDX models, but with the **best instrumental** that I could get.
(Remembers : It's for **Karaoke !**).

I also added a few tricks to improve the quality of the results.

# 🚀 INSTALLATION

You can run KaraFan with any [Frontends that support Jupyter widgets](https://github.com/jupyter/jupyter/wiki/Jupyter-Widgets#frontends-that-support-jupyter-widgets)

I develop this project with and it works on :

✅ Google Colab

Go there :

and copy this Colab notebook on your Google Drive.  
Then open it with Google Colab.

**IMPORTANT** :
If the system disconnects while saving an audio file, you will need to delete this file before restarting Colab, as the saved file may be incomplete !!
And especially if the "**GOD Mode**" is activated !
.

✅ Your PC with Visual Studio **Code**

1️⃣ Install the Jupyter extension from Microsoft in VS Code

2️⃣ Create a new folder on your PC where you wish to store the « KaraFan » project.

- You can choose any folder you want, but it's better to choose the same folder where you store your Google Drive.
- For example, on my PC : I have a folder named « Mon Drive » (in french) that is synchronized with my Google Drive.

3️⃣ Clone this repository in this folder by using the command line :

```bash
git clone https://github.com/Captain-FLAM/KaraFan.git
```

Or if you don't have git installed, you can download the zip file and unzip it in the folder you created.

4️⃣ Go inside the folder « KaraFan » and execute the following command to install all the required Python packages :

- Windows :

```bash
Install.bat
```

- Linux / Mac :

```bash
py -3.10 -m App.setup
or
python -m App.setup
```

   For Linux/Mac, install PyTorch CUDA (take a look at the [PyTorch website](https://pytorch.org/get-started/locally/) for more details)
.

✅ Your PC with a command line in a shell :

Follow the same steps as for Visual Studio Code, but without the step 1️⃣.

Example of usage :

```bash
python -m App.inference --help
```

```bash
python -m App.inference -i song1.mp3 song2.flac song3.wav --overlap_MDX 0.25 --chunk_size 500000
```

# 💡 TECHNICAL DETAILS

## « My Magic Recipe »

| Step                                                | Filename                   |
| --------------------------------------------------- | -------------------------- |
| 1 - Normalization of Original audio                 | 0 - NORMALIZED.flac        |
| 2 - Instrumental Extraction from Normalized         | 1 - Music_extract.flac     |
| 3 - Volume Compensation for Instrumental            | (internal)                 |
| 4 - Subtraction of Instrumental from Normalized     | 2 - Audio_sub_Music.flac   |
| 5 - Vocal Extraction from cleaned "Audio_sub_Music" | 3 - Vocals.flac            |
| 6 - Volume Compensation for Vocals                  | (internal)                 |
| 7 - Subtraction of Vocals from Normalized           | 4 - Music.flac             |
| 8 - Bleeding Vocals/Other in final "Music"          | 5 - Bleeding_in_Music.flac |

Details of each step :

 1️⃣ **Normalization of Original audio**

- Normalize audio to -1.0 db peak amplitude

This is mandatory because every process is based on RMS db levels.
(Volumes Compensations & audio Substractions)

2️⃣ **Instrumental Extraction from Normalized**
You will understand that I only use this model to extract the instrumental part to have at most a clean vocals, but it is not used in the final result.

- Use the model to isolate the instrumental parts of the audio track.

3️⃣ **Volume Compensation for Instrumental**

- Internal step involving volume compensation for the extracted instrumental.

4️⃣ **Subtraction of Instrumental from Normalized**
The instrumental part is then subtracted from the previously normalized to obtain an audio track with only vocals.

- Isolate the vocal parts.

5️⃣ **Vocal Extraction from cleaned "Audio_sub_Music"**

- Use the model to isolate the vocal component of the music track, removing any remaining instrumental or background noise.

6️⃣ **Volume Compensation for Vocals**

- Internal step involving volume compensation for the extracted vocal audio.

7️⃣ **Subtraction of Vocals from Normalized**
The vocal parts are subtracted from the previously normalized to obtain an audio track with only instrumental music.

- Isolate the instrumental component from the original audio normalized.

8️⃣ **Bleeding Vocals/Other in "Music"**
The bleeding vocals or other elements are obtained by subtracting 1st "Music_extract" track from the final "Music" track.

- Obtain an audio track that contains any residual vocal or other elements present in the final instrumental music.

These steps collectively represent the audio processing workflow, which separates vocals and instruments from a music track and handles various audio adjustments and filtering.
Some steps involve internal operations without generating separate output files.

#### « Volume Compensation »

These are **very important values** that needs to be **fine-tuned for each model**, to obtain the best results.

Volume compensation is a process that adjusts the volume of the audio to compensate for the volume changes that occur during the separation process. This is necessary because the volume of the audio is reduced during the separation process. The volume compensation process is performed internally and does not generate a separate output file.

#### About « Silent » filter

Make silent the parts of audio where dynamic range (RMS) goes below threshold.
Don't misundertand : this function is NOT a noise reduction !
Its behavior is to clean the audio from "silent parts" (below -50 dB) to :

- avoid the MLM model to work on "silent parts", and save GPU time
- avoid the MLM model to produce artifacts on "silent parts"
- clean the final Vocals audio files from residues of "silent parts" (and get back them in "Music")

## « SRS » - Soprano mode by Jarredou

Option to use the soprano mode as a model bandwidth extender to make narrowband models fullband. (At least those with a cutoff at 17.5khz).

**Description of the trick :**

* process the input audio at original sample rate
* process the input audio with shifted sample rate by a ratio that make the original audio spectrum fit in the model bandwidth, then restore the original samplerate
* use lowpass & highpass filters to create the multiband ensemble of the 2 separated audio, using the shifted sample rate results as the high band to fill what's above the cutoff of the model.
* with scipy.signal.resample_poly, a ratio of 5/4 for up/down before processing does the trick for models with cutoff at 17.5khz*

**User Stories/Use Cases/Benefits:**

Fullband results with "old" narrowband models

**Potential Challenges/Considerations:**

A smooth transition with zerophase soft filtering between the 2 bands works better than brickwall filters, around 14000hz was a good value in my few tests.
Make sure to not have volume changes in the crossover region (I've used Linkwitz-Riley filters).

Downside is first, the doubled separation time because of the 2 passes, and that the separation quality of the shifted sample rate audio is often lower than the normal processed one, but in most of the cases, as it's using only its high freq, it's enough the make that "fullband trick" works very well !
.

# 💗 SPECIAL THANKS TO...

All these wonderful people who have indirectly contributed to the realization of this project :

✔️ [Jarredou]()  
✔️ [deton24]()  
✔️ [MVSep.com]()  

✔️ [Kuielab]()  
✔️ [Anjok07]()  
✔️ [Kimberley Jensen]()  
✔️ [Facebook Research]()  

We are all standing on the shoulders of giants !
.

# 🦄 IN THE NEAR FUTURE

You will see ...
I have hundreds of ideas, but I need time to implement them.

And as I'm an independent developer, I have to work to earn my living.

So if you want to encourage me to give more time & improve this project, you can make a donation or become one of my patrons :

* [![img](https://img.shields.io/badge/Donate-Bitcoin-yellow.svg)]()
* [![img](https://img.shields.io/badge/Donate-Buy%20me%20a%20beer-yellow.svg)]()
* [![img](https://img.shields.io/badge/Donate-Patreon-red.svg)]()

This helps me to :

- Pay calculation time on Google Colab (100 units = 11 €, I eat all in 3-4 days)
- or to buy new hardware to make more and more tests to improve the quality of the results.
  I'm currently using a 4 years old laptop with a GTX 1060 Ti 4GB
  I'm dreaming of a RTX 4090 ... and I'm NOT a gamer !
  ... and maybe if you are too genereous, Me can buy an Nvidia A100 40GB !! 😍

.

# 📝 TODO

Stupid ideas given by Copilot : (I'm not responsible for this !) 😆

- [ ] Add a « Karaoke » mode (with lyrics)
- [ ] Add a « Music » mode (without lyrics)
- [ ] Add a « Instrumental » mode (without vocals)
- [ ] Add a « Acapella » mode (without instruments)
- [ ] Add a « Remix » mode (with a different mix)

..., and now, my brain is overheating ! 😵

- [ ] Add more Models (but always with fine-tuned volume compensation !)
- [ ] Separate Choirs from Vocals (very hard challenge !)
- [ ] etc ...

# 📜 LICENSE

### &copy; Captain FLAM - 2023 - MIT license

That means you can do whatever you want with this code, but **you have to** mention my name and the fact that I'm the original author of this code, and mention the names of all the people who have contributed to this project.

**You have to** keep the original license file in your project, and keep the original header with copyrights in each source file.

---
