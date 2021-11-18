#Run the StatisticsGen component: 

 

from tfx.components import StatisticsGen  

statistics_gen = StatisticsGen( 

 examples=example_gen.outputs['examples']) 

context.run(statistics_gen) 

context.show(statistics_gen.outputs['statistics']) 

 

#Run the Schema Gen component: 

 

from tfx.components import SchemaGen  

schema_gen = SchemaGen( 

 statistics=statistics_gen.outputs['statistics'], 

 infer_feature_shape=True) 

context.run(schema_gen) 

 

#Creating the example validator component taht takes as input the schema and statistics: 

 

from tfx.components import ExampleValidator  

example_validator = ExampleValidator( 

 statistics=statistics_gen.outputs['statistics'], 

 schema=schema_gen.outputs['schema']) 

context.run(example_validator) 

 