
import scipy.stats as stats

ALL_DIST = ('UNIFORM', 'LOG_UNIFORM')

def sample_tuning(config):
    ''' Given a config dictionary, sample variables based on given distribution '''

    # sample each variable 
    results = {}
    for key, var_dict in config.items():
        
        # check if distribution is implemented
        dist = var_dict['dist']
        if dist not in ALL_DIST:
            raise ValueError('{} sampling dist {} is not implemented.'.format(key, dist))
        
        # sample and add to result dict
        if dist == 'UNIFORM':
            rvar = stats.randint.rvs(var_dict['min'], var_dict['max'])
        elif dist == 'LOG_UNIFORM':
            rvar = stats.loguniform.rvs(var_dict['min'], var_dict['max'])
        results[key] = rvar

    return results

