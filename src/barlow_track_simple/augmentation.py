from typing import Tuple

import numpy.typing as npt
import torchio.transforms as tio


class Transform:
    def __init__(
        self,
        p_RandomAffine_base=1.0,
        p_RandomBlur_base=0.1,
        p_RandomBlur=0.0,
        p_RandomAffine=0.1,
        p_RandomAffine_flip=0.1,
        zxy_RandomElasticDeformation=(1, 3, 3),
        p_RandomElasticDeformation=0.0,
        std_RandomNoise=0.25,
        p_RandomNoise=0.1,
        p_RandomAffine_both=None,
        **kwargs,  # Catch unknown args
    ):

        if isinstance(p_RandomAffine_both, float):
            p_RandomAffine_base = p_RandomAffine_both
            p_RandomAffine_flip = p_RandomAffine_both

        # This normalization should get rid of the noise floor (~100) and keep the actual peak values
        self.final_normalization = tio.RescaleIntensity(percentiles=(5, 100))
        # self.final_normalization_no_copy = tio.RescaleIntensity(
        #     percentiles=(5, 100), copy=False
        # )

        self.transform = tio.Compose(
            [
                tio.RandomAffine(degrees=(180, 0, 0), p=p_RandomAffine_base),
                tio.RandomBlur(p=p_RandomBlur_base),
                self.final_normalization,
            ]
        )
        self.transform_prime = tio.Compose(
            [
                # tio.RandomFlip(axes=(1, 2), p=args.get('p_RandomFlip', 0.0)),  # Do not flip z
                tio.RandomBlur(p=p_RandomBlur),
                tio.RandomAffine(
                    degrees=(180, 0, 0),
                    p=p_RandomAffine,
                ),
                tio.RandomAffine(
                    degrees=(180, 180, 0, 0, 0, 0),
                    p=p_RandomAffine_flip,
                ),  # A 180 degree rotation, like a flip
                tio.RandomElasticDeformation(
                    max_displacement=zxy_RandomElasticDeformation,
                    p=p_RandomElasticDeformation,
                ),
                tio.RandomNoise(
                    std=std_RandomNoise,
                    p=p_RandomNoise,
                ),
                # tio.ZNormalization()
                self.final_normalization,
            ]
        )

        print("Initialized Transformation class with augmentation and probabilities:")
        print([(_t.name, _t.probability) for _t in self.transform])
        print([(_t.name, _t.probability) for _t in self.transform_prime])

    def __call__(self, x: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        y1 = self.transform(x)  # type: ignore
        y2 = self.transform_prime(x)  # type: ignore
        return y1, y2

    def normalize(self, img: npt.ArrayLike) -> npt.ArrayLike:
        return self.final_normalization(img)  # type: ignore
