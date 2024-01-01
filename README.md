# EECS598-QuantumSystemArchitecture
 Project repo for F23' EECS598-QuantumSystemArchitecture

 ### Prerequisite
* Python 3.7 or above
* matplotlib 3.2.1
* numpy 1.18.4
* SciPy 1.4.0
* Shapely 1.7.1 
* NetworkX 2.6.2
* PyQt5
* pyqtgraph
* h5py
* networkx-metis

### Installation Guide
```python
  pip install -r requirements.txt
```
#### METIS
The [networkx-metis package](https://github.com/networkx/networkx-metis/) requres extra care
* Step 1: [Cython](https://github.com/cython/cython/wiki/CythonExtensionsOnWindows)
```python
pip install Cython
```
* Step 2: Git clone to 'lib/TrapGeometry'

```sh
git clone https://github.com/networkx/networkx-metis.git
cd networkx-metis

```
* Step 3: Change header files (might be risky)
`networkx-metis\src\GKlib\gk_arch.h [line 63]` annotate the code `#define rint(x) ((int)((x)+0.5))`.
`networkx-metis\src\GKlib\ms_stdint.h` replace all
`int8_t` to `ms_int8_t`,
`int_least8_t` to `ms_int_least8_t`,
`int_fast8_t` to `ms_int_fast8_t`,
`int_fast16_t` to `ms_int_fast16_t`,
`uint_fast16_t` to `ms_uint_fast16_t`.

* Step 4: Make and Build
```python
python setup.py build --compiler=msvc
python setup.py install
```
