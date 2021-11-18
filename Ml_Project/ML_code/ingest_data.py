#Import the TF record file created using ImportExampleGen which represented by the example gen: 

 

import os  

from tfx.components import ImportExampleGen 

from tfx.utils.dsl_utils import external_input 

base_dir = os.getcwd() 

data_dir = os.path.join(os.pardir, "tfrecord_data") 

examples = external_input(os.path.join('/content/drive/MyDrive/','training data')) 

example_gen = ImportExampleGen(input=examples) 

context.run(example_gen)  # runing the example gen component