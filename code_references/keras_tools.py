import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Multiply, Add, Activation
from keras.models import Model
from tensorflow.keras import regularizers
# Custom activation function
from keras import backend as K
import keras_tuner as kt
### same model as built above but now ready for Bayesian 

def res_block(inputs, norm_type, activation, dropout, ff_dim, n_dense, initializer, regularizer):
    """Residual block of TSMixer."""

    norm = (
      keras.layers.LayerNormalization
      if norm_type == 'L'
      else keras.layers.BatchNormalization
    )

    # Temporal Linear
    x = norm(axis=[-2, -1])(inputs)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = Dense(x.shape[-1], kernel_initializer = initializer, kernel_regularizer=regularizer, activation=activation)(x)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
    x = keras.layers.Dropout(dropout)(x)
    res = x + inputs

    # Feature Linear
    x = norm(axis=[-2, -1])(res)
    for _ in range(n_dense):
        x = Dense(ff_dim,kernel_initializer = initializer, kernel_regularizer=regularizer, activation=activation)(x)  # [Batch, Input Length, FF_Dim]
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
    x = keras.layers.Dropout(dropout)(x)
    return x + res

def build_VT(layers_VT, units_VT, n_dense_VT, input_layer, input_mult, initializer, regularizer, dropout):
    
    hidden = input_layer
    for i in range(layers_VT):
         hidden = res_block(hidden, 'L', 'selu', dropout, units_VT, n_dense_VT ,initializer, regularizer)

    hidden = hidden[:,:,0]
    #hidden = tf.transpose(hidden, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    output = keras.layers.Dense(1, kernel_initializer = initializer, kernel_regularizer=regularizer, activation = 'exponential', name=f'eVNN_layer_{i+1}')(hidden) 
  
    multiply_output = keras.layers.Multiply()([output, input_mult])
    
    shaped_output = keras.layers.Reshape((3,1))(multiply_output)
    
    model = keras.Model(inputs = [input_layer, input_mult], outputs = [shaped_output])
    
    return model

def build_TBNN(layers_TBNN, units_TBNN, n_dense_TBNN, input_layer, input_tensor_basis, initializer, regularizer, dropout):
    # Build the hidden layers
    hidden = input_layer
    for i in range(layers_TBNN):
         hidden = res_block(hidden, 'L', 'selu', dropout, units_TBNN, n_dense_TBNN, initializer, regularizer)
            
    hidden = hidden[:,:,0]
    #The layer of gn, which are coefficients for each of the ten Tn
    output = keras.layers.Dense(20, kernel_initializer = initializer, kernel_regularizer=regularizer, activation = 'selu', name=f'TBNN_layer_{i+1}')(hidden) 
 
    #Multiply the gn by Tn, with the output being the anisotropy tensor
    shaped = keras.layers.Reshape((20,1,1))(output)
    merge = keras.layers.Dot(axes=1)([shaped, input_tensor_basis])
    #Reshape the output anisotropy tensor, and trim out duplicate values (it is a symmetric matrix). The end result is a 6 component vector.
    shaped_output = keras.layers.Reshape((9,1),name='Shaped_output')(merge)

    model = keras.Model(inputs = [input_layer, input_tensor_basis], outputs = [shaped_output])
    
    return model

### Tensor Based Neural Network for the  components of b
def build_Framework(lr, layers_VT, units_VT, n_dense_VT, layers_TBNN, units_TBNN, n_dense_TBNN, cwd, current_file, dropout_VT=0, dropout_TBNN=0):
    #3 inputs: 
    # set of features for learning
    # set of tensors with working as invariant basis
    # shat to predict anisotropy
    input_layer = keras.layers.Input(shape=(15,3), name = 'input_layer')
    input_tensor_basis = keras.layers.Input(shape=(20,3,3), name = 'tensor_input_layer')
    input_Shat = keras.layers.Input(shape=(3), name = 'input_Shat')
    
    tf.random.set_seed(84)
    
    #Select kernel intializer and regularizer
    initializer = tf.keras.initializers.LecunNormal(seed=0)
    regularizer = regularizers.L2(1e-8)
    
    #Define the loss function
    loss_fn = tf.keras.losses.Huber()

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    
    if layers_VT == None and units_VT == None:
        visc_model = tf.keras.models.load_model(f'{cwd}\\models\\saved_model_{current_file}_eVNN.h5')
        visc_model._name = "Pre_Trained_eVNN"
        visc_model.trainable = False
        print(visc_model.summary())
    else:
        visc_model = build_VT(layers_VT, units_VT, n_dense_VT, input_layer, input_Shat, initializer, regularizer, dropout_VT)
        visc_model._name = "eVNN"

    if layers_TBNN == None and units_TBNN == None:
        TBNN_model = tf.keras.models.load_model(f'C:\\Users\\GL504GS\\Desktop\\TBNN\\Compiled Cases\\1.TBNN\\_3b_ff\\models\\saved_model_base_TBNN_3b_f{current_file[-1]}.h5').get_layer(model_dict[current_file[-1]])
        TBNN_model._name = "Pre_Trained_TBNN"
        #TBNN_model.trainable = False
    else:
        TBNN_model = build_TBNN(layers_TBNN, units_TBNN, n_dense_TBNN, input_layer, input_tensor_basis, initializer, regularizer, dropout_TBNN)
        TBNN_model._name = "TBNN"
    
    #tke_TBNN = build_VT(input_layer, TBNN_model([input_layer, input_tensor_basis]), initializer, regularizer)
    
    # multiply the coefficient corresponding to vt by Shat to 
    # to calculate de linear part of anisotropy
    shaped_output_visc = visc_model([input_layer, input_Shat]) 
    
    trimmed_output_visc0 = keras.layers.Lambda(lambda x : -x[:,0])(shaped_output_visc)
    trimmed_output_visc1 = keras.layers.Lambda(lambda x : -x[:,1])(shaped_output_visc)
    trimmed_output_visc4 = keras.layers.Lambda(lambda x : -x[:,2])(shaped_output_visc) 
    
    
    ### multiply the coefficients generated by the tensor basis and add the turbulent kinetic term
    shaped_output_TBNN = TBNN_model([input_layer, input_tensor_basis])
        
    trimmed_output0 = keras.layers.Lambda(lambda x : x[:,0])(shaped_output_TBNN)
    trimmed_output1 = keras.layers.Lambda(lambda x : x[:,1])(shaped_output_TBNN)
    trimmed_output4 = keras.layers.Lambda(lambda x : x[:,4])(shaped_output_TBNN)

    ### add linear and non linear terms together
    trimmed_output0 = keras.layers.Add()([trimmed_output0, trimmed_output_visc0])
    trimmed_output1 = keras.layers.Add()([trimmed_output1, trimmed_output_visc1])
    trimmed_output4 = keras.layers.Add()([trimmed_output4, trimmed_output_visc4])
        
    trimmed_output6 = keras.layers.Add()([trimmed_output0, trimmed_output4])
    
    merged_output = tf.keras.layers.Concatenate()([trimmed_output0, trimmed_output1, trimmed_output4, tf.math.negative(trimmed_output6)])

    
    model = keras.Model(inputs = [input_layer, input_tensor_basis, input_Shat], outputs = [merged_output])
       
    model.compile(loss = loss_fn, optimizer = optimizer, metrics = ['mse', 'mae'])
    
    model._name = f"eVTBNN_Framework_{current_file[-2:]}"
    
    print(model.summary())
    
    return model





def eVNN_block(x, layers_VT, units_VT,  initializer, regularizer):
    hidden = x
    for i in range(layers_VT):
        hidden = Dense(units = units_VT, kernel_initializer = initializer, kernel_regularizer=regularizer, activation = 'selu')(hidden) # name=f'eVNN_layer_{i}'
    #Both turbulent viscosity and tke are allways poistive so an exponential activation enforces that   
    output = Dense(1, kernel_initializer = initializer, kernel_regularizer=regularizer, activation = 'selu')(hidden) # , name=f'eVNN_layer_{i+1}'
    return x + output

def custom_tanh(x):
    return 1.5*K.tanh(x)+.5

def build_eVNN(lr, layers_eVNN, units_eVNN, n_dense_eVNN, cwd, current_file):
    #3 inputs:
    # set of features for learning
    # set of tensors with working as invariant basis
    # shat to predict anisotropy
    input_layer = keras.layers.Input(shape=(15,3), name = 'input_layer')
    input_Shat = keras.layers.Input(shape=(3), name = 'input_Shat')
    
    tf.random.set_seed(42)
    
    #Select kernel intializer and regularizer
    initializer = tf.keras.initializers.LecunNormal(seed=0)
    regularizer = regularizers.L2(1e-8)
    
    #Define the loss function
    loss_fn = tf.keras.losses.Huber()

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    
    hidden = input_layer

    # eV_units = 128
    # for i in range(8):
    #     hidden = Dense(units = eV_units, kernel_initializer = initializer, kernel_regularizer=regularizer, activation = 'selu')(hidden)
    #     eV_units *= 0.5


    for i in range(layers_eVNN):
       hidden = res_block(hidden, 'L', 'selu', 0, units_eVNN, n_dense_eVNN, initializer, regularizer)

       #eVNN_block(x, layers_VT, units_VT,  initializer, regularizer)

    output = hidden[:,:,0]

    output = Dense(1, kernel_initializer = initializer, kernel_regularizer=regularizer, activation = 'exponential', name='eV_final_value')(output) # , name=f'eVNN_layer_{i+1}'

    output = -2*output
    
    multiply_output = keras.layers.Multiply()([output, input_Shat])
    
    shaped_output = keras.layers.Reshape((3,1))(multiply_output)

    model = keras.Model(inputs = [input_layer, input_Shat], outputs = [shaped_output])
       
    model.compile(loss = loss_fn, optimizer = optimizer, metrics = ['mse', 'mae'])
    
    model._name = "eVNN"
    
    print(model.summary())
    
    return model


# def build_eVNN(lr, layers_VT, units_VT, cwd, current_file):
#     #3 inputs:
#     # set of features for learning
#     # set of tensors with working as invariant basis
#     # shat to predict anisotropy
#     input_layer = keras.layers.Input(shape=(15), name = 'input_layer')
    
#     tf.random.set_seed(42)
    
#     #Select kernel intializer and regularizer
#     initializer = tf.keras.initializers.LecunNormal(seed=0)
#     regularizer = regularizers.L2(1e-8)
    
#     #Define the loss function
#     loss_fn = tf.keras.losses.Huber()

#     # Define the optimizer
#     optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    
#     hidden = input_layer
#     for i in range(layers_VT):
#         hidden = Dense(units = units_VT, kernel_initializer = initializer, kernel_regularizer=regularizer, activation = 'selu', name=f'eVNN_layer_{i}')(hidden)
    
#     #Both turbulent viscosity and tke are allways poistive so an exponential activation enforces that   
#     output = Dense(1, kernel_initializer = initializer, kernel_regularizer=regularizer, activation = 'exponential', name=f'eVNN_layer_{i+1}')(hidden)
    
#     model = keras.Model(inputs = [input_layer], outputs = [output])
       
#     model.compile(loss = loss_fn, optimizer = optimizer, metrics = ['mse', 'mae'])
    
#     model._name = "eVNN"
    
#     #print(model.summary())
    
#     return model


def opt_model(hp):
    units_VT = hp.Int("units_VT", min_value=150, max_value=300, step=10)
    units_TBNN = hp.Int("units_TBNN", min_value=150, max_value=300, step=10)
    
    layers_VT = hp.Int("layers_VT", min_value=20, max_value=100, step=5)
    layers_TBNN = hp.Int("layers_TBNN", min_value=20, max_value=100, step=5)
    
    lr = hp.Float("lr", min_value=1e-8, max_value=5e-4, sampling="log")
    # call existing model-building code with the hyperparameter values.
    framework = build_Framework(lr = lr, layers_VT = layers_VT, units_VT = units_VT, layers_TBNN = layers_TBNN, units_TBNN = units_TBNN)
    return framework

def opt_model(hp):
    #units_VT = hp.Int("units_VT", min_value=150, max_value=300, step=10)
    units_TBNN = hp.Int("units_TBNN", min_value=150, max_value=300, step=10)
    
    #layers_VT = hp.Int("layers_VT", min_value=20, max_value=100, step=5)
    layers_TBNN = hp.Int("layers_TBNN", min_value=20, max_value=100, step=5)
    
    lr = hp.Float("lr", min_value=1e-8, max_value=5e-4, sampling="log")
    # call existing model-building code with the hyperparameter values.
    framework = build_Framework(lr = lr, layers_TBNN = layers_TBNN, units_TBNN = units_TBNN)
    return framework


def start_opt(opt_model, current_file):
    tuner = kt.tuners.BayesianOptimization(
                hypermodel = opt_model,
                seed = 0,
                objective = 'val_loss',
                max_trials = 10,
                executions_per_trial = 2,
                directory = 'Bayes_Opt',
                project_name = f'tuning_{current_file}')
    return tuner

def extract_eVNN(model):
    if type(model) == list:
        eVNN = [Model(model_k.get_layer('eVNN').layers[0].input, model_k.get_layer('eVNN').layers[-4].output) for model_k in model]   
    else:
        _eVNN = model.get_layer('eVNN')
        #_eVNN = _eVNN.get_layer('eVNN')
        eVNN = Model(_eVNN.layers[0].input, _eVNN.layers[-4].output)
        print(eVNN.summary())
    return eVNN
