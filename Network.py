# this script only build networks

from tensorflow.keras.layers import Dense,concatenate,Input,add
from tensorflow.keras.layers import Activation,Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D,Conv2D, Conv2DTranspose,MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam


# Residual block
def res_block_gen(model, kernal_size, filters, strides):
    gen = model
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = add([gen, model])
    return model
    

def discriminator_block(model, filters, kernel_size, strides):
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model


def network_(image_shape,nb_class):
    #build generator network------------------------------------
    input_1 = Input(shape = image_shape)

    out = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(input_1)
    out = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(out)
     
    for index in range(9): ## residual 
        out = res_block_gen(out, 3, 64, 1)

    out = discriminator_block(out, 128, 3, 2)
    out = discriminator_block(out, 128, 3, 2)
    out = discriminator_block(out, 256, 3, 2)
    out = discriminator_block(out, 256, 3, 2)

    out = Flatten()(out)
    out = LeakyReLU(alpha = 0.2)(out)
    out = Dense(nb_class)(out)
    out = Activation('softmax')(out) 

    cnn = Model(inputs = input_1, outputs = out)#zursan networkoo negdtgeed neg model bolgox
    # done -----------------------------------------------------
    
    #initialize two networks
    optimizer = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    cnn.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics = ['accuracy'])
    
    print('cnn structure :')
    print(cnn.summary())

    return cnn
