from preprocessing import *
from postprocessing import *
from model import *

if __name__ == '__main__':
    exp_list = ['20210609_cytox_preculture_run2', '20210609_cytox_preculture_run1',
                '20210610_cytox_preculture_run2', '20210610_cytox_preculture_run1',
                '20210611_cytox_preculture_run2', '20210611_cytox_preculture_run1']

    res = dict([['20210609_cytox_preculture_run1C0C0', ['A', 'B', 'F']],
                ['20210609_cytox_preculture_run2C0C0', ['A', 'B', 'F']],
                ['20210610_cytox_preculture_run1C0C0', ['A', 'B', 'E']],
                ['20210610_cytox_preculture_run2C0C0', ['A', 'B', 'E']],
                ['20210611_cytox_preculture_run1C0C0', ['C', 'D', 'H']],
                ['20210611_cytox_preculture_run2C0C0', ['C', 'D', 'H']]])

    sus = dict([['20210609_cytox_preculture_run1C0C0', 'E'],
                ['20210609_cytox_preculture_run2C0C0', 'E'],
                ['20210610_cytox_preculture_run1C0C0', 'F'],
                ['20210610_cytox_preculture_run2C0C0', 'F'],
                ['20210611_cytox_preculture_run1C0C0', 'G'],
                ['20210611_cytox_preculture_run2C0C0', 'G']])

    perfect_shuffle('/Users/eugenemiller/Desktop/Mod2_71621/Experiments',
                    '/Users/eugenemiller/Desktop/Mod2_71621/ModelDat',
                    res_wells=res, sus_wells=sus)

    generate_roc_CNN('/Users/eugenemiller/Desktop/Mod2_71621/ModelDat/train',
                     '/Users/eugenemiller/Desktop/Mod2_71621/ModelDat/valid',
                     '/Users/eugenemiller/Desktop/Mod2_71621/ModelDat/test')

    model = tf.keras.models.load_model('/Users/eugenemiller/Desktop/Mod2_71621/saved')
    color_all_preds('/Users/eugenemiller/Desktop/savearr_test', model)