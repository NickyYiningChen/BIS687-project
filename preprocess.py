import numpy as np
from tqdm import tqdm

import os
import time
import json
import argparse
from glob import glob


import argparse
import time
import os

def setup_arguments():
    argument_parser = argparse.ArgumentParser(description='preprocessing help')
    argument_parser.add_argument('--data-dir', type=str, default='data/processed', help='directory for processed data')
    return argument_parser.parse_args()

def mkdir(d):
    path = d.split('/')
    for i in range(len(path)):
        d = '/'.join(path[:i+1])
        if not os.path.exists(d):
            os.mkdir(d)


def csv_split(line, sc=','):
    res = []
    inside = 0
    s = ''
    for c in line:
        if inside == 0 and c == sc:
            res.append(s)
            s = ''
        else:
            if c == '"':
                inside = 1 - inside
            s = s + c
    res.append(s)
    return res


def convert_timestamp(timestamp):
    try:
        timestamp = float(timestamp)
        return timestamp
    except:
        timestamp = str(timestamp).replace('"', '')
        epoch_time = time.mktime(time.strptime(timestamp, '%Y-%m-%d %H:%M:%S'))
        hours_since_epoch = int(epoch_time / 3600)
        return hours_since_epoch

def create_patient_files(config, feature_file_path):
    feature_selection = []
    patient_data_dir = config.initial_dir
    os.system('rm -r ' + patient_data_dir)
    mkdir(patient_data_dir)
    for line_index, line_content in enumerate(open(feature_file_path)):
        if line_index % 10000 == 0:
            print(line_index)
        if line_index:
            elements = line_content.strip().split(',')

            assert len(elements) == len(column_headers)
            selected_data = [elements[idx] for idx in feature_selection]
            patient_data = ','.join(selected_data)

            patient_file_path = os.path.join(patient_data_dir, elements[0] + '.csv')
            if not os.path.exists(patient_file_path):
                file_writer = open(patient_file_path, 'w')
                file_writer.write(column_header_line)
                file_writer.close()
            file_writer = open(patient_file_path, 'a')
            file_writer.write('\n' + patient_data)
            file_writer.close()
        else:
            column_headers = csv_split(line_content.strip())
            column_headers = [header.strip('"') for header in column_headers]
            print('There are {:d} features.'.format(len(column_headers)))
            print(column_headers)
            if len(feature_selection) == 0:
                feature_selection = range(1, len(column_headers))
                selected_headers = [column_headers[index].replace('"', '').replace(',', ';') for index in feature_selection]
                column_header_line = ','.join(selected_headers)


def aggregate_patient_intervals(config, time_frame=1, time_threshold=-48):
    aggregate_directory = config.resample_dir
    data_source_directory = config.initial_dir

    os.system('rm -r ' + aggregate_directory)
    mkdir(aggregate_directory)

    interval_counts = [0, 0]
    interval_frequency = dict()
    patient_group_sets = [set(), set()]
    for file_index, file_name in enumerate(tqdm(os.listdir(data_source_directory))):
        patient_time_dict = dict()
        for line_index, line_content in enumerate(open(os.path.join(data_source_directory, file_name))):
            if line_index:
                if len(line_content.strip()) == 0:
                    continue
                content_elements = line_content.strip().split(',')
                assert len(content_elements) == len(header_names)
                current_time = convert_timestamp(content_elements[0])
                rounded_time = time_frame * int(float(current_time) / time_frame)
                if rounded_time not in patient_time_dict:
                    patient_time_dict[rounded_time] = []
                patient_time_dict[rounded_time].append(content_elements)
            else:
                header_names = line_content.strip().split(',')
                header_names[0] = 'time'

        output_file = open(os.path.join(aggregate_directory, file_name), 'w')
        output_file.write(','.join(header_names))
        previous_time = None
        visible = 0
        maximum_time = max(patient_time_dict)
        for time_key in sorted(patient_time_dict):
            if time_key - maximum_time < time_threshold:
                continue
            record_list = patient_time_dict[time_key]
            merged_record = record_list[0]
            for record_elements in record_list:
                for element_index, value in enumerate(record_elements):
                    if len(value.strip()):
                        merged_record[element_index] = value
            merged_record[0] = str(time_key - maximum_time)
            merged_line = '\n' + ','.join(merged_record)
            output_file.write(merged_line)

            if previous_time is not None:
                time_difference = time_key - previous_time
                if time_difference > time_frame:
                    visible = 1
                    interval_counts[0] += 1
                    interval_frequency[time_difference] = interval_frequency.get(time_difference, 0) + 1
                    patient_group_sets[0].add(file_name)
                patient_group_sets[1].add(file_name)
                interval_counts[1] += 1
            previous_time = time_key
        output_file.close()
    print('There are {:d}/{:d} collections data with intervals > {:d}.'.format(interval_counts[0], interval_counts[1], time_frame))
    print('There are {:d}/{:d} patients with intervals > {:d}.'.format(len(patient_group_sets[0]), len(patient_group_sets[1]), time_frame))



def compile_feature_statistics(settings):
    aggregated_data_dir = settings.resample_dir
    file_paths = sorted(glob(os.path.join(aggregated_data_dir, '*')))
    feature_value_dict = dict()
    feature_missing_dict = dict()
    for file_index, file_path in enumerate(tqdm(file_paths)):
        if 'csv' not in file_path:
            continue
        for line_index, line_content in enumerate(open(file_path)):
            line_content = line_content.strip()
            if line_index == 0:
                feat_list = line_content.split(',')
            else:
                values = line_content.split(',')
                for value_index, value in enumerate(values):
                    if value in ['NA', '']:
                        continue
                    field_name = feat_list[value_index]
                    if field_name not in feature_value_dict:
                        feature_value_dict[field_name] = []
                    feature_value_dict[field_name].append(float(value))

    feature_mm_dict = dict()
    feature_ms_dict = dict()
    feature_range_dict = dict()
    length_time_series = max([len(value_list) for value_list in feature_value_dict.values()])
    for field, value_list in feature_value_dict.items():
        sorted_values = sorted(value_list)
        value_intervals = []
        for interval_index in range(settings.split_num):
            position = int(interval_index * len(sorted_values) / settings.split_num)
            value_intervals.append(sorted_values[position])
        value_intervals.append(sorted_values[-1])
        feature_range_dict[field] = value_intervals

        position = int(len(sorted_values) / settings.split_num)
        feature_mm_dict[field] = [sorted_values[position], sorted_values[-position - 1]]
        feature_ms_dict[field] = [np.mean(sorted_values), np.std(sorted_values)]

        feature_missing_dict[field] = 1.0 - float(len(value_list)) / length_time_series

    json.dump(feature_mm_dict, open(os.path.join(settings.files_dir, 'feature_mm_dict.json'), 'w'))
    json.dump(feature_ms_dict, open(os.path.join(settings.files_dir, 'feature_ms_dict.json'), 'w'))
    json.dump(feat_list, open(os.path.join(settings.files_dir, 'feature_list.json'), 'w'))
    json.dump(feature_missing_dict, open(os.path.join(settings.files_dir, 'feature_missing_dict.json'), 'w'))
    json.dump(feature_range_dict, open(os.path.join(settings.files_dir, f'feature_value_dict_{settings.split_num}.json'), 'w'))



def partition_data_into_deciles(settings):
    data_directory = settings.resample_dir
    data_files = sorted(glob(os.path.join(data_directory, '*')))
    np.random.shuffle(data_files)
    decile_splits = []
    for decile in range(10):
        start_index = int(len(data_files) * decile / 10)
        end_index = int(len(data_files) * (decile + 1) / 10)
        decile_splits.append(data_files[start_index:end_index])
    json.dump(decile_splits, open(os.path.join(settings.files_dir, 'splits.json'), 'w'))

def map_labels_to_ids(settings, data_type):
    label_mapping = dict()
    for line_index, line_content in enumerate(open(os.path.join(settings.data_dir, f'{data_type}.csv'))):
        if line_index:
            entries = line_content.strip().split(',')
            patient_id = entries[0]
            if patient_id in label_mapping and label_mapping[patient_id] != 0:
              continue
            patient_label = ''.join(entries[1:])
            patient_id = str(int(float(patient_id)))
            label_mapping[patient_id] = patient_label
    json.dump(label_mapping, open(os.path.join(settings.files_dir, f'{data_type}_dict.json'), 'w'))

def compile_demographics(settings, demographics_file):
    demographics_map = dict()
    demographics_index = dict()
    for line_index, line_content in enumerate(open(demographics_file)):
        if line_index:
            entries = line_content.strip().split(',')
            patient_id = str(int(float(entries[0])))
            demographics_map[patient_id] = []
            for demographic in entries[1:]:
                if demographic not in demographics_index:
                    demographics_index[demographic] = len(demographics_index)
                demographics_map[patient_id].append(demographics_index[demographic])
    json.dump(demographics_map, open(os.path.join(settings.files_dir, 'demo_dict.json'), 'w'))
    json.dump(demographics_index, open(os.path.join(settings.files_dir, 'demo_index_dict.json'), 'w'))

def main():
    config = setup_arguments()
    config.files_dir = os.path.join(config.data_dir, 'files')
    config.initial_dir = os.path.join(config.data_dir, 'initial_data')
    config.resample_dir = os.path.join(config.data_dir, 'resample_dir')
    config.split_num = 4000
    mkdir(config.files_dir)
    mkdir(config.initial_dir)
    mkdir(config.resample_dir)
    features_csv_path = os.path.join(config.data_dir, 'features.csv')
    demographics_csv_path = os.path.join(config.data_dir, 'demo.csv')
    for task in ['mortality', 'readmit']:
        map_labels_to_ids(config, task)
    compile_demographics(config, demographics_csv_path)
    create_patient_files(config, features_csv_path)
    aggregate_patient_intervals(config)
    compile_feature_statistics(config)
    partition_data_into_deciles(config)

if __name__ == '__main__':
    main()
