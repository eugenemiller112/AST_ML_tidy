# commonly used experiments/paths for ease
exp_list = ['20210609_cytox_preculture_run2', '20210609_cytox_preculture_run1',
                '20210610_cytox_preculture_run2', '20210610_cytox_preculture_run1',
                '20210611_cytox_preculture_run2', '20210611_cytox_preculture_run1']

res = dict([['20210609_cytox_preculture_run1', ['A', 'B', 'F']],
                ['20210609_cytox_preculture_run2', ['A', 'B', 'F']],
                ['20210610_cytox_preculture_run1', ['A', 'B', 'E']],
                ['20210610_cytox_preculture_run2', ['A', 'B', 'E']],
                ['20210611_cytox_preculture_run1', ['C', 'D', 'H']],
                ['20210611_cytox_preculture_run2', ['C', 'D', 'H']]])

sus = dict([['20210609_cytox_preculture_run1', 'E'],
                ['20210609_cytox_preculture_run2', 'E'],
                ['20210610_cytox_preculture_run1', 'F'],
                ['20210610_cytox_preculture_run2', 'F'],
                ['20210611_cytox_preculture_run1', 'G'],
                ['20210611_cytox_preculture_run2', 'G']])

#/Volumes/External/20220325_KanTitration

jitter_settings = {'lag': 3,
                   'crop': 100,
                   'upsample': 100}

seg_settings = {'crop': 200,
                'min_sigma': 30,
                'max_sigma': 50,
                'num_sigma': 50,
                'threshold': .01,
                'overlap': 0,
                'radius': 5,
                'min_size': 200,
                'block_size': 3}