import cv2
import albumentations as A

class HorizontalFlipKeepIndexOrder(A.HorizontalFlip):
    """Flip the input horizontally but keypoint the same keypoints indices."""

    def apply_to_keypoints(self, keypoints, **params):
        flipped_keypoints = [self.apply_to_keypoint(tuple(keypoint[:4]), **params) + tuple(keypoint[4:]) for keypoint in keypoints]
        flipped_keypoints.reverse()
        return flipped_keypoints

def load_transform(augmentation_type: str, augmentation_probability: float, input_size: int = 256, num_frames: int =1) ->  A.core.composition.Compose:

    crop_ratio =1.0 #0.78125

    if augmentation_type == "strongkeep":
        input_transformer = A.Compose([
            HorizontalFlipKeepIndexOrder(p=0.5),
            A.OneOf(
                [
                    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=45,
                                       interpolation=cv2.INTER_NEAREST, p=1.0),  # 0.25, 0.10, 0.50
                    A.RandomSizedCrop(min_max_height=(int(crop_ratio*input_size), int(crop_ratio*input_size)), height=input_size, width=input_size, p=1.00), #0.75),
                ],
                p=1.00
            ),
            A.OneOf(
                [
                    # Apply one of transforms to 50% of images
                    A.RandomBrightnessContrast(),
                    A.RandomBrightness(),  # Apply random brightness
                ],
                p=0.5
            ),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
            additional_targets={'teacher_kpts': 'keypoints'},
            p=augmentation_probability)

    elif augmentation_type == "strongkeep_echo":
        input_img_size = 112
        input_transformer = A.Compose([
            HorizontalFlipKeepIndexOrder(p=0.5),
            A.OneOf(
                [
                    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=45,
                                       interpolation=cv2.INTER_NEAREST, p=1.0),  # 0.25, 0.10, 0.50
                    A.RandomSizedCrop(min_max_height=(int(crop_ratio*input_img_size), int(crop_ratio*input_img_size)),
                                      height=input_img_size, width=input_img_size, p=1.00),  # 0.75),
                ],
                p=1.00
            ),
            A.OneOf(
                [
                    # Apply one of transforms to 50% of images
                    A.RandomBrightnessContrast(),  # Apply random brightness and contrast
                    A.RandomGamma(), # Apply random gamma

                ],
                p=0.5
            ),

        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    elif augmentation_type == "strong_echo_cycle":
        input_img_size = 112
        additionaltargets = {}
        #todo change to dynamic number of images
        num_images = num_frames
        for i in range(1, num_images):
            additionaltargets= {**additionaltargets, f'image{i}': 'image'}
        additionaltargets['keypoints1']='keypoints'
        #additionaltargets['keypoints2']='keypoints'

        input_transformer = A.Compose([HorizontalFlipKeepIndexOrder(p=0.5),
            A.OneOf(
                [
                    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=45,
                                       interpolation=cv2.INTER_NEAREST, p=1.0),  # 0.25, 0.10, 0.50
                    A.RandomSizedCrop(min_max_height=(int(crop_ratio*input_img_size), int(crop_ratio*input_img_size)),
                                      height=input_img_size, width=input_img_size, p=1.00),  # 0.75),
                ],
                p=1.00
            ),
            A.OneOf(
                [
                    # Apply one of transforms to 50% of images
                    A.RandomBrightnessContrast(),  # Apply random brightness and contrast
                    A.RandomGamma(),  # Apply random gamma
                ],
                p=0.5
            ),
        ],
            additional_targets=additionaltargets,
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


    else:
        raise NotImplementedError("Augmentation method is currently not implemented..")

    return input_transformer


if __name__ == '__main__':

    input_size = 224
    augmentation_type = "2chkeep"
    input_transform = None
    input_transform = load_transform(augmentation_type=augmentation_type, augmentation_probability=1.0, input_size=input_size)
