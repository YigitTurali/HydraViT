import os
import numpy as np
import random
import tensorflow as tf
from keras.models import Model
from keras import layers

from keras.layers import Dense, Flatten, Dropout, Embedding, Input, Layer, LayerNormalization, MultiHeadAttention, Add, Multiply,Concatenate, Subtract
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from typing import Optional, Tuple, List
from matplotlib import pyplot as plt
from random import randint


seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)



isimler_train2 = np.load('isimler_train_3_son.npy')
isimler_test2 = np.load('isimler_test_3_son.npy')
etiketler_train2 = np.load('etiketler_train_3_son.npy')
etiketler_test2 = np.load('etiketler_test_3_son.npy')





AUTO = tf.data.AUTOTUNE


b_size = 35
epo = 120
projection_dim = 512
num_heads = 4
patch_size = 3
image_size =7
num_patches = (image_size // patch_size) ** 2
transformer_units = [projection_dim * 2, projection_dim]  # Size of the transformer layers
transformer_layers = 4
mlp_head_units = [25088]  # Size of the dense layers of the final classifier
DROP_PATH_RATE = 0.1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
opt = tf.optimizers.Adam(learning_rate=0.0001)


dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range=10)
dataGen.fit(isimler_train2)



class PatchEmbed(layers.Layer):
    """Image patch embedding layer, also acts as the down-sampling layer.

    Args:
        image_size (Tuple[int]): Input image resolution.
        patch_size (Tuple[int]): Patch spatial resolution.
        embed_dim (int): Embedding dimension.
    """

    def __init__(
        self,
        image_size: Tuple[int] = (image_size, image_size),
        patch_size: Tuple[int] = (patch_size, patch_size),
        embed_dim: int = projection_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        patch_resolution = [
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        ]
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_resolution = patch_resolution
        self.num_patches = patch_resolution[0] * patch_resolution[1]
        self.proj = layers.Conv2D(
            filters=embed_dim, kernel_size=patch_size, strides=patch_size
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))
        self.norm = layers.LayerNormalization(epsilon=1e-7)

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, int, int, int]:
        """Patchifies the image and converts into tokens.

        Args:
            x: Tensor of shape (B, H, W, C)

        Returns:
            A tuple of the processed tensor, height of the projected
            feature map, width of the projected feature map, number
            of channels of the projected feature map.
        """
        # Project the inputs.
        x = self.proj(x)

        # Obtain the shape from the projected tensor.
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = tf.shape(x)[3]

        # B, H, W, C -> B, H*W, C
        x = self.norm(self.flatten(x))

        return x, height, width, channels


def MLP(
    in_features: int,
    hidden_features: Optional[int] = None,
    out_features: Optional[int] = None,
    mlp_drop_rate: float = 0.0,
):
    hidden_features = hidden_features or in_features
    out_features = out_features or in_features

    return tf.keras.Sequential(
        [
            layers.Dense(units=hidden_features, activation=tf.keras.activations.gelu),
            layers.Dense(units=out_features),
            layers.Dropout(rate=mlp_drop_rate),
        ]
    )




class FocalModulationLayer(layers.Layer):
    """The Focal Modulation layer includes query projection & context aggregation.

    Args:
        dim (int): Projection dimension.
        focal_window (int): Window size for focal modulation.
        focal_level (int): The current focal level.
        focal_factor (int): Factor of focal modulation.
        proj_drop_rate (float): Rate of dropout.
    """

    def __init__(
        self,
        dim: int,
        focal_window: int,
        focal_level: int,
        focal_factor: int = 2,
        proj_drop_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.proj_drop_rate = proj_drop_rate

        # Project the input feature into a new feature space using a
        # linear layer. Note the `units` used. We will be projecting the input
        # feature all at once and split the projection into query, context,
        # and gates.
        self.initial_proj = layers.Dense(
            units=(2 * self.dim) + (self.focal_level + 1),
            use_bias=True,
        )
        self.focal_layers = list()
        self.kernel_sizes = list()
        for idx in range(self.focal_level):
            kernel_size = (self.focal_factor * idx) + self.focal_window
            depth_gelu_block = tf.keras.Sequential(
                [
                    layers.ZeroPadding2D(padding=(kernel_size // 2, kernel_size // 2)),
                    layers.Conv2D(
                        filters=self.dim,
                        kernel_size=kernel_size,
                        activation=tf.keras.activations.gelu,
                        groups=self.dim,
                        use_bias=False,
                    ),
                ]
            )
            self.focal_layers.append(depth_gelu_block)
            self.kernel_sizes.append(kernel_size)
        self.activation = tf.keras.activations.gelu
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.modulator_proj = layers.Conv2D(
            filters=self.dim,
            kernel_size=(1, 1),
            use_bias=True,
        )
        self.proj = layers.Dense(units=self.dim)
        self.proj_drop = layers.Dropout(self.proj_drop_rate)

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            x: Tensor of shape (B, H, W, C)
        """
        # Apply the linear projecion to the input feature map
        x_proj = self.initial_proj(x)

        # Split the projected x into query, context and gates
        query, context, self.gates = tf.split(
            value=x_proj,
            num_or_size_splits=[self.dim, self.dim, self.focal_level + 1],
            axis=-1,
        )

        # Context aggregation
        context = self.focal_layers[0](context)
        context_all = context * self.gates[..., 0:1]
        for idx in range(1, self.focal_level):
            context = self.focal_layers[idx](context)
            context_all += context * self.gates[..., idx : idx + 1]

        # Build the global context
        context_global = self.activation(self.gap(context))
        context_all += context_global * self.gates[..., self.focal_level :]

        # Focal Modulation
        self.modulator = self.modulator_proj(context_all)
        x_output = query * self.modulator

        # Project the output and apply dropout
        x_output = self.proj(x_output)
        x_output = self.proj_drop(x_output)

        return x_output





class FocalModulationBlock(layers.Layer):
    """Combine FFN and Focal Modulation Layer.

    Args:
        dim (int): Number of input channels.
        input_resolution (Tuple[int]): Input resulotion.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float): Dropout rate.
        drop_path (float): Stochastic depth rate.
        focal_level (int): Number of focal levels.
        focal_window (int): Focal window size at first focal level
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int],
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        focal_level: int = 1,
        focal_window: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.modulation = FocalModulationLayer(
            dim=self.dim,
            focal_window=self.focal_window,
            focal_level=self.focal_level,
            proj_drop_rate=drop,
        )
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(
            in_features=self.dim,
            hidden_features=mlp_hidden_dim,
            mlp_drop_rate=drop,
        )

    def call(self, x: tf.Tensor, height: int, width: int, channels: int) -> tf.Tensor:
        """Processes the input tensor through the focal modulation block.

        Args:
            x (tf.Tensor): Inputs of the shape (B, L, C)
            height (int): The height of the feature map
            width (int): The width of the feature map
            channels (int): The number of channels of the feature map

        Returns:
            The processed tensor.
        """
        shortcut = x

        # Focal Modulation
        x = tf.reshape(x, shape=(-1, height, width, channels))
        x = self.modulation(x)
        x = tf.reshape(x, shape=(-1, height * width, channels))

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm(x))
        return x



class BasicLayer(layers.Layer):
    """Collection of Focal Modulation Blocks.

    Args:
        dim (int): Dimensions of the model.
        out_dim (int): Dimension used by the Patch Embedding Layer.
        input_resolution (Tuple[int]): Input image resolution.
        depth (int): The number of Focal Modulation Blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float): Dropout rate.
        downsample (tf.keras.layers.Layer): Downsampling layer at the end of the layer.
        focal_level (int): The current focal level.
        focal_window (int): Focal window used.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        input_resolution: Tuple[int],
        depth: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        downsample=None,
        focal_level: int = 1,
        focal_window: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = [
            FocalModulationBlock(
                dim=dim,
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio,
                drop=drop,
                focal_level=focal_level,
                focal_window=focal_window,
            )
            for i in range(self.depth)
        ]

        # Downsample layer at the end of the layer
        if downsample is not None:
            self.downsample = downsample(
                image_size=input_resolution,
                patch_size=(2, 2),
                embed_dim=out_dim,
            )
        else:
            self.downsample = None

    def call(
        self, x: tf.Tensor, height: int, width: int, channels: int
    ) -> Tuple[tf.Tensor, int, int, int]:
        """Forward pass of the layer.

        Args:
            x (tf.Tensor): Tensor of shape (B, L, C)
            height (int): Height of feature map
            width (int): Width of feature map
            channels (int): Embed Dim of feature map

        Returns:
            A tuple of the processed tensor, changed height, width, and
            dim of the tensor.
        """
        # Apply Focal Modulation Blocks
        for block in self.blocks:
            x = block(x, height, width, channels)

        # Except the last Basic Layer, all the layers have
        # downsample at the end of it.
        if self.downsample is not None:
            x = tf.reshape(x, shape=(-1, height, width, channels))
            x, height_o, width_o, channels_o = self.downsample(x)
        else:
            height_o, width_o, channels_o = height, width, channels

        return x, height_o, width_o, channels_o




class FocalModulationNetwork(Model):
    """The Focal Modulation Network.

    Parameters:
        image_size (Tuple[int]): Spatial size of images used.
        patch_size (Tuple[int]): Patch size of each patch.
        num_classes (int): Number of classes used for classification.
        embed_dim (int): Patch embedding dimension.
        depths (List[int]): Depth of each Focal Transformer block.
        mlp_ratio (float): Ratio of expansion for the intermediate layer of MLP.
        drop_rate (float): The dropout rate for FM and MLP layers.
        focal_levels (list): How many focal levels at all stages.
            Note that this excludes the finest-grain level.
        focal_windows (list): The focal window size at all stages.
    """



    def __init__(
        self,
        image_size: Tuple[int] = (image_size, image_size),
        patch_size: Tuple[int] = (patch_size, patch_size),

        embed_dim: int = projection_dim,
        depths: List[int] = [2, 3, 2],
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        focal_levels=[2, 2, 2],
        focal_windows=[3, 3, 3],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = len(depths)
        embed_dim = [embed_dim * (2**i) for i in range(self.num_layers)]

        self.embed_dim = embed_dim
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim[0],
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patch_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = layers.Dropout(drop_rate)
        self.basic_layers = list()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dim[i_layer],
                out_dim=embed_dim[i_layer + 1]
                if (i_layer < self.num_layers - 1)
                else None,
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                focal_level=focal_levels[i_layer],
                focal_window=focal_windows[i_layer],
            )
            self.basic_layers.append(layer)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-7)
        self.avgpool = layers.GlobalAveragePooling1D()
        self.flatten = layers.Flatten()


    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            x: Tensor of shape (B, H, W, C)

        Returns:
            The logits.
        """
        # Patch Embed the input images.
        x, height, width, channels = self.patch_embed(x)
        x = self.pos_drop(x)

        for idx, layer in enumerate(self.basic_layers):
            x, height, width, channels = layer(x, height, width, channels)

        x = self.norm(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x




# Some code is taken from:
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2.
class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super().__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)
        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )















def HydraNet(gamma):
    inputs = Input(shape=(224, 224, 3), name='input')
    hydra_body = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))(inputs)

    encoder_output = FocalModulationNetwork()(hydra_body)



    class_0_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_0_output1")(
        encoder_output)
    class_1_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_1_output1")(
        encoder_output)
    class_2_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_2_output1")(
        encoder_output)
    class_3_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_3_output1")(
        encoder_output)
    class_4_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_4_output1")(
        encoder_output)
    class_5_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_5_output1")(
        encoder_output)
    class_6_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_6_output1")(
        encoder_output)
    class_7_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_7_output1")(
        encoder_output)
    class_8_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_8_output1")(
        encoder_output)
    class_9_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_9_output1")(
        encoder_output)
    class_10_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_10_output1")(
        encoder_output)
    class_11_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_11_output1")(
        encoder_output)
    class_12_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_12_output1")(
        encoder_output)
    class_13_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_13_output1")(
        encoder_output)
    class_14_branch1 = Dense(units=1, kernel_initializer='he_uniform', activation="sigmoid", name="task_14_output1")(
        encoder_output)
    distance_combined_branch1 = Dense(units=15, kernel_initializer='he_uniform', activation="sigmoid", name="distance_combined_output1")(
        encoder_output)





    model = Model(inputs=inputs,
                  outputs=[class_0_branch1, class_1_branch1, class_2_branch1, class_3_branch1, class_4_branch1,
                           class_5_branch1, class_6_branch1, class_7_branch1, class_8_branch1, class_9_branch1,
                           class_10_branch1, class_11_branch1, class_12_branch1, class_13_branch1, class_14_branch1,
                           distance_combined_branch1])

    model.summary()
    return model




def compile_hydranet_model(model, gamma, optimizer):



    model.compile(optimizer=optimizer,
                  loss={'task_0_output': 'binary_crossentropy',
                        'task_1_output': 'binary_crossentropy',
                        'task_2_output': 'binary_crossentropy',
                        'task_3_output': 'binary_crossentropy',
                        'task_4_output': 'binary_crossentropy',
                        'task_5_output': 'binary_crossentropy',
                        'task_6_output': 'binary_crossentropy',
                        'task_7_output': 'binary_crossentropy',
                        'task_8_output': 'binary_crossentropy',
                        'task_9_output': 'binary_crossentropy',
                        'task_10_output': 'binary_crossentropy',
                        'task_11_output': 'binary_crossentropy',
                        'task_12_output': 'binary_crossentropy',
                        'task_13_output': 'binary_crossentropy',
                        'task_14_output': 'binary_crossentropy',
                        'distance_combined_output': 'binary_crossentropy',
                        },
                  loss_weights={
                      'task_0_output': (gamma[0])[0][0],
                      'task_1_output': (gamma[1])[0][0],
                      'task_2_output': (gamma[2])[0][0],
                      'task_3_output': (gamma[3])[0][0],
                      'task_4_output': (gamma[4])[0][0],
                      'task_5_output': (gamma[5])[0][0],
                      'task_6_output': (gamma[6])[0][0],
                      'task_7_output': (gamma[7])[0][0],
                      'task_8_output': (gamma[8])[0][0],
                      'task_9_output': (gamma[9])[0][0],
                      'task_10_output': (gamma[10])[0][0],
                      'task_11_output': (gamma[11])[0][0],
                      'task_12_output': (gamma[12])[0][0],
                      'task_13_output': (gamma[13])[0][0],
                      'task_14_output': (gamma[14])[0][0],
                      'distance_combined_output': (gamma[15])[0][0],
                  },
                  metrics=['accuracy'])
    return model




def make_multi_output_flow(image_gen, X, y_list, batch_size):
    y_item_0 = y_list[0]
    y_indices = np.arange(y_item_0.shape[0])
    orig_flow = image_gen.flow(X, y=y_indices, batch_size=batch_size)
    while True:
        (X, y_next_i) = next(orig_flow)
        y_next = [y_item[y_next_i] for y_item in y_list]
        yield X, y_next




def batch_fit(gamma_values):

    hist = list()
    trained_models = list()
    y_train = [etiketler_train2[:, 0],
               etiketler_train2[:, 1],
               etiketler_train2[:, 2],
               etiketler_train2[:, 3],
               etiketler_train2[:, 4],
               etiketler_train2[:, 5],
               etiketler_train2[:, 6],
               etiketler_train2[:, 7],
               etiketler_train2[:, 8],
               etiketler_train2[:, 9],
               etiketler_train2[:, 10],
               etiketler_train2[:, 11],
               etiketler_train2[:, 12],
               etiketler_train2[:, 13],
               etiketler_train2[:, 14],
               etiketler_train2[:, :15]
               ]


    model = HydraNet(gamma_values)




    import tensorflow_addons as tfa

    optimizer = tfa.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)


    model = compile_hydranet_model(model, gamma_values,optimizer)




    model_hist = model.fit(make_multi_output_flow(dataGen, isimler_train2, y_train, b_size),
                           epochs=epo, verbose='auto',
                           steps_per_epoch=isimler_train2.shape[0] // b_size)







    hist.append(model_hist)
    trained_models.append(model)

    ans = isimler_test2
    ans = ans.reshape(-1, 224, 224, 3)
    proba2 = model.predict(ans)

    proba = proba2[0:15]
    proba = np.array(proba)
    proba = np.squeeze(proba)
    proba = np.transpose(proba)
    filepath2 = "predict_VGG_HydraNet_with_bfl.npy"
    np.save(filepath2, proba)

    proba3 = proba2[15]
    filepath2 = "predict_VGG_HydraNet_with_bfl_2.npy"
    np.save(filepath2, proba3)


    return hist, trained_models



total_steps = int((len(isimler_train2) / b_size) * epo)
warmup_epoch_percentage = 0.15
warmup_steps = int(total_steps * warmup_epoch_percentage)
scheduled_lrs = WarmUpCosine(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=0.0,
    warmup_steps=warmup_steps,
)

hist, trained_models = batch_fit( [0.0184528, 0.07715134, 0.04684685, 0.09059233, 0.01546698, 0.08441558, 0.12807882, 1.0, 0.01030111, 0.03667137,
     0.00341431, 0.03346203, 0.06582278, 0.13541667, 0.04113924, 0.06])










