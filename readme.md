# prepare data
put the data into ./data
then run 
```python
python ./tools.py
```
# generate the noise labels
run 
```python
python ./noise_label/generate.py
```
# run the code
run 
```python
python ./train.py --gpus=0
```