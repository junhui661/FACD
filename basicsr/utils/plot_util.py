import re


def read_data_from_tensorboard(log_path, tag):
                                                               

         
                                                    
                                  
       
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

                       
    event_acc = EventAccumulator(log_path)
    event_acc.Reload()
    scalar_list = event_acc.Tags()['scalars']
    print('tag list: ', scalar_list)
    steps = [int(s.step) for s in event_acc.Scalars(tag)]
    values = [s.value for s in event_acc.Scalars(tag)]
    return steps, values


def read_data_from_txt_2v(path, pattern, step_one=False):
                                                                         

         
                                         
                                                       
                                                        
       
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    steps = []
    values = []

    pattern = re.compile(pattern)
    for line in lines:
        match = pattern.match(line)
        if match:
            steps.append(int(match.group(1)))
            values.append(float(match.group(2)))
    if step_one:
        steps = [v + 1 for v in steps]
    return steps, values


def read_data_from_txt_1v(path, pattern):
                                                 

         
                                         
                                                       
       
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    data = []

    pattern = re.compile(pattern)
    for line in lines:
        match = pattern.match(line)
        if match:
            data.append(float(match.group(1)))
    return data


def smooth_data(values, smooth_weight):
                                                                               

                                                                                                                                                                                   

         
                                                       
                                             
       
    values_sm = []
    last_sm_value = values[0]
    for value in values:
        value_sm = last_sm_value * smooth_weight + (1 - smooth_weight) * value
        values_sm.append(value_sm)
        last_sm_value = value_sm
    return values_sm
