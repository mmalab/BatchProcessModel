from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

def dir_gen(config):
        '''flow from directory
        # Return
            train_datagenerator:    image generator for training
            val_datagenerator:      image generator for validation
        '''
        # train data must be applied DataAugmentation
        train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                           rotation_range=config.da_rotate,
                                           width_shift_range=config.da_width,
                                           height_shift_range=config.da_height,
                                           shear_range=config.da_shear,
                                           zoom_range=config.da_zoom,
                                           samplewise_std_normalization=config.da_std,
                                           channel_shift_range=config.da_channel,
                                           horizontal_flip=config.da_hflip,
                                           vertical_flip=config.da_vflip)
        # validation data mustn't be applied DataAugmentation
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        ## make generator from directory
        ### for validation
        train_datagenerator = train_datagen.flow_from_directory(
            Path(config.data_path, 'train').as_posix(),
            target_size = config.img_size,
            batch_size = config.batch_size,
            class_mode=config.gen_class,
            color_mode=config.gen_color
        )
        ### for training
        val_datagenerator = val_datagen.flow_from_directory(
            Path(config.data_path, 'validation').as_posix(),
            target_size = config.img_size,
            batch_size = config.batch_size,
            class_mode=config.gen_class,
            color_mode=config.gen_color
        )

        return train_datagenerator, val_datagenerator
