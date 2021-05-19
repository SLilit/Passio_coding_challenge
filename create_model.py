import tensorflow as tf

def create_mobilenetv2_model(inputs, output_dim = 1280):    
    
    base_model = tf.keras.applications.MobileNetV2(input_shape = (224, 224, 3),
                                                   include_top = False,
                                                   weights = 'imagenet')
    base_model.trainable = False
    x = base_model(inputs)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Dense(units = output_dim, activation = 'relu')(x)
    outputs = tf.math.l2_normalize(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model


inputs = tf.keras.Input(shape=(224, 224, 3))
mobilenetv2_model = create_mobilenetv2_model(inputs)
mobilenetv2_model.save('model/mobilenetv2_model.h5')
