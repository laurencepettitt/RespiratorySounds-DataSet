### Description

#### Context

Respiratory sounds are important indicators of respiratory health and respiratory disorders. The sound emitted when a person breathes is directly related to air movement, changes within lung tissue and the position of secretions within the lung. A wheezing sound, for example, is a common sign that a patient has an obstructive airway disease like asthma or chronic obstructive pulmonary disease (COPD).

These sounds can be recorded using digital stethoscopes and other recording techniques. This digital data opens up the possibility of using machine learning to automatically diagnose respiratory disorders like asthma, pneumonia and bronchiolitis, to name a few.

#### Content

The Respiratory Sound Database was created by two research teams in Portugal and Greece. It includes 920 annotated recordings of varying length - 10s to 90s. These recordings were taken from 126 patients. There are a total of 5.5 hours of recordings containing 6898 respiratory cycles - 1864 contain crackles, 886 contain wheezes and 506 contain both crackles and wheezes. The data includes both clean respiratory sounds as well as noisy recordings that simulate real life conditions. The patients span all age groups - children, adults and the elderly.

This Kaggle dataset includes:

    920 .wav sound files
    920 annotation .txt files
    A text file listing the diagnosis for each patient
    A text file explaining the file naming format
    A text file listing 91 names (filename_differences.txt )
    A text file containing demographic information for each patient

#### General

```
 The demographic info file has 6 columns:
  - Patient number
  - Age
  - Sex
  - Adult BMI (kg/m2)
  - Child Weight (kg)
  - Child Height (cm)

Each audio file name is divided into 5 elements, separated with underscores (_).

1. Patient number (101,102,...,226)
2. Recording index
3. Chest location 
     a. Trachea (Tc)
     b. Anterior left (Al)
     c. Anterior right (Ar)
     d. Posterior left (Pl)
     e. Posterior right (Pr)
     f. Lateral left (Ll)
     g. Lateral right (Lr)
4. Acquisition mode 
     a. sequential/single channel (sc), 
     b. simultaneous/multichannel (mc)
5. Recording equipment 
     a. AKG C417L Microphone (AKGC417L), 
     b. 3M Littmann Classic II SE Stethoscope (LittC2SE), 
     c. 3M Litmmann 3200 Electronic Stethoscope (Litt3200), 
     d.  WelchAllyn Meditron Master Elite Electronic Stethoscope (Meditron)

The annotation text files have four columns:
- Beginning of respiratory cycle(s)
- End of respiratory cycle(s)
- Presence/absence of crackles (presence=1, absence=0)
- Presence/absence of wheezes (presence=1, absence=0)

The abbreviations used in the diagnosis file are:
- COPD: Chronic Obstructive Pulmonary Disease
- LRTI: Lower Respiratory Tract Infection
- URTI: Upper Respiratory Tract Infection
```

#### Citation

###### Paper
Α Respiratory Sound Database for the Development of Automated Classification
Rocha BM, Filos D, Mendes L, Vogiatzis I, Perantoni E, Kaimakamis E, Natsiavas P, Oliveira A, Jácome C, Marques A, Paiva RP (2018) In Precision Medicine Powered by pHealth and Connected Health (pp. 51-55). Springer, Singapore.
[https://eden.dei.uc.pt/%7Eruipedro/publications/Conferences/ICBHI2017a.pdf](https://eden.dei.uc.pt/%7Eruipedro/publications/Conferences/ICBHI2017a.pdf)

Ref Websites

- [http://www.auditory.org/mhonarc/2018/msg00007.html](http://www.auditory.org/mhonarc/2018/msg00007.html)
- [http://bhichallenge.med.auth.gr/](http://www.auditory.org/mhonarc/2018/msg00007.html)

#### Acknowledgements

Many thanks to the research teams at the University of Coimbra, Portugal; the University de Aveiro, Portugal and the Aristotle University of Thessaloniki, Greece for making this dataset publicly available.