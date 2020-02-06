import os
import logging
from io import BytesIO

import numpy as np
import PIL
import svgwrite
from cairosvg.parser import Tree
from cairosvg.surface import PNGSurface

from handwriting_synthesis import BASE_PATH, drawing
from handwriting_synthesis.rnn import rnn


class Hand(object):

    def __init__(self, checkpoint_dir, prediction_dir):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.nn = rnn(
            log_dir='logs',
            checkpoint_dir=checkpoint_dir,
            prediction_dir=prediction_dir,
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

    def write(self, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None, scales=None, xscales=None, yscales=None):
        valid_char_set = set(drawing.alphabet)

        if scales is not None and (xscales is not None or yscales is not None):
            raise ValueError('Cannot provide a flat scaling value in addition to individual axis scaling.')

        for line_num, line in enumerate(lines):
            if len(line) > 75:
                raise ValueError(
                    (
                        "Each line must be at most 75 characters. "
                        "Line {} contains {}"
                    ).format(line_num, len(line))
                )

            for char in line:
                if char not in valid_char_set:
                    raise ValueError(
                        (
                            "Invalid character {} detected in line {}. "
                            "Valid character set is {}"
                        ).format(char, line_num, valid_char_set)
                    )

        strokes = self._sample(lines, biases=biases, styles=styles)
        return self._draw(strokes, lines, stroke_colors=stroke_colors, stroke_widths=stroke_widths, scales=scales, xscales=xscales, yscales=yscales)

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40*max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5]*num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load(os.path.join(BASE_PATH, 'styles/style-{}-strokes.npy'.format(style)))
                c_p = np.load(os.path.join(BASE_PATH, 'styles/style-{}-chars.npy'.format(style))).tostring().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples

    def _draw(self, strokes, lines, stroke_colors=None, stroke_widths=None, scales=None, xscales=None, yscales=None):

        stroke_colors = stroke_colors or ['black']*len(lines)
        stroke_widths = stroke_widths or [2]*len(lines)
        scales = scales or [1.0]*len(lines)
        xscales = xscales or [1.0]*len(lines)
        yscales = yscales or [1.0]*len(lines)

        max_xsc = max(xscales + scales)
        max_ysc = max(yscales + scales)

        line_height = 60
        view_width, view_height = 1000, line_height*(len(lines) + 1)

        dwg = svgwrite.Drawing()
        dwg.viewbox(width=view_width*max_xsc, height=view_height*max_ysc)
        dwg.add(dwg.rect(insert=(0, 0), size=(view_width*max_xsc, view_height*max_ysc), fill='white'))

        initial_coord = np.array([0.4*view_width, -0.5*line_height])
        for offsets, line, color, width, scale, xsc, ysc in zip(strokes, lines, stroke_colors, stroke_widths, scales, xscales, yscales):

            if not line:
                initial_coord[1] -= line_height
                continue

            offsets[:, :2] *= 1.5 * scale
            strokes = drawing.offsets_to_coords(offsets)
            strokes = drawing.denoise(strokes)
            strokes[:, :2] = drawing.align(strokes[:, :2])

            strokes[:, 1] *= -1
            strokes[:, :2] -= strokes[:, :2].min() + initial_coord
            strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2

            prev_eos = 1.0
            p = "M{},{} ".format(0, 0)
            for x, y, eos in zip(*strokes.T):
                x *= xsc
                y *= ysc
                p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
                prev_eos = eos
            path = svgwrite.path.Path(p)
            path = path.stroke(color=color, width=width, linecap='round').fill("none")
            dwg.add(path)

            initial_coord[1] -= line_height

        surface = PNGSurface(
            tree=Tree(bytestring=dwg.tostring(), unsafe=False),
            output=BytesIO(),
            dpi=600
        )
        surface.finish()
        return PIL.Image.open(surface.output)
