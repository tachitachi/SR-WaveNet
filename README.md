# SR-WaveNet

Download nsynth dataset (json version) from: 

```
https://magenta.tensorflow.org/datasets/nsynth#files
```

And extract into a folder located at:

```
path/to/repository/nsynth_data
```

Create tfrecord file with:

```
python create_tfrecord.py
```

Train teacher network with:

```
python teacher.py --train
```


Train student network with:

```
python student.py --train --teacher path/to/teacher/checkpoint/dir
```
