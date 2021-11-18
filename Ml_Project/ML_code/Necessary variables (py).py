
#Create the context variable that allow you to run each pipeline component separetly by runing context.run(component) 

 

import tensorflow as tf  

from tfx.orchestration.experimental.interactive.interactive_context import \ 

 InteractiveContext 

context = InteractiveContext() 

 

#Import all the pipeline component we will be using: 

 

import os  

import pprint 

import tempfile 

import urllib 

 

import absl 

import tensorflow as tf 

import tensorflow_model_analysis as tfma 

tf.get_logger().propagate = False 

pp = pprint.PrettyPrinter() 

 

import tfx 

from tfx.components import CsvExampleGen 

from tfx.components import Evaluator 

from tfx.components import ExampleValidator 

from tfx.components import Pusher 

from tfx.components import ResolverNode 

from tfx.components import SchemaGen 

from tfx.components import StatisticsGen 

from tfx.components import Trainer 

from tfx.components import Transform 

from tfx.dsl.experimental import latest_blessed_model_resolver 

from tfx.orchestration import metadata 

from tfx.orchestration import pipeline 

from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext 

from tfx.proto import pusher_pb2 

from tfx.proto import trainer_pb2 

from tfx.proto.evaluator_pb2 import SingleSlicingSpec 

from tfx.utils.dsl_utils import external_input 

from tfx.types import Channel 

from tfx.types.standard_artifacts import Model 

from tfx.types.standard_artifacts import ModelBlessing  
