from scipy import stats
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import numbers as nums

FILE_PATH = 'file.txt'


def count_avg(numbers):
    return float(sum(numbers)) / len(numbers)


def count_skew_crit(n):
    ans = 3 * (6 * (n - 1) / ((n + 1) * (n + 3))) ** 0.5
    return ans


def count_kurtosis_crit(n):
    ans = 5 * (24 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))) ** 0.5
    return ans


class NormaList:
    def __init__(self, raw_list: list):
        self.raw_list = raw_list

        self.average = 0
        self.n = 0

        self.np_array = 0

        self.standart_deviation = 0

        self.skew = 0
        self.skew_crit = 0

        self.kurtosis = 0
        self.kurtosis_crit = 0

        self.recount()

    def recount(self):
        self.average = count_avg(self.raw_list)
        self.n = len(self.raw_list)

        self.np_array = np.array(self.raw_list)

        self.standart_deviation = (self.np_array.std())

        self.skew = stats.skew(self.np_array)
        self.skew_crit = count_skew_crit(self.n)

        self.kurtosis = stats.kurtosis(self.np_array)
        self.kurtosis_crit = count_kurtosis_crit(self.n)

    def __getitem__(self, key):
        return self.raw_list[key]

    def __setitem__(self, key, value):
        self.raw_list[key] = value

    def __delitem__(self, key):
        del self.raw_list[key]

    def __iter__(self):
        return iter(self.raw_list)

    def __len__(self):
        return len(self.raw_list)

    def __str__(self, *args, **kwargs):
        return str(self.raw_list)


class Normalizer:

    def __init__(self, x_list, y_list):
        self.x_list = NormaList(x_list)
        self.y_list = NormaList(y_list)

    def make_normal(self):
        if not (self.check_is_normal(self.x_list) and self.check_is_normal(self.y_list)):

            result_1, result_2 = True, True

            while result_1 is True or result_2 is True:

                result_1 = self.delete_anomal(self.x_list, self.y_list)
                result_2 = self.delete_anomal(self.y_list, self.x_list)

                self.x_list.recount()
                self.y_list.recount()

            print("\nВсе аномалии были удалены\n")

    @staticmethod
    def delete_anomal(normal_list_1: NormaList, normal_list_2: NormaList):
        anything_deleted = False
        for i in range(len(normal_list_1)):
            if normal_list_1[i] > normal_list_1.average + 3 * normal_list_1.standart_deviation or \
                    normal_list_1[i] < normal_list_1.average - 3 * normal_list_1.standart_deviation:
                print(f'{normal_list_1[i]} - аномалия\n')
                del normal_list_1[i]
                del normal_list_2[i]
                anything_deleted = True
                break
        return anything_deleted

    @staticmethod   
    def check_is_normal(normal_list: NormaList):
        if normal_list.skew < normal_list.skew_crit and normal_list.kurtosis < normal_list.kurtosis_crit:
            return True
        else:
            return False

    def get_normal_lists(self):
        return self.x_list, self.y_list


class Pearson:
    def __init__(self, x_list: NormaList, y_list: NormaList):
        self.x_list = x_list
        self.y_list = y_list
        self.pearson, self.pearson_pvalue = stats.pearsonr(x_list, y_list)

    def check_zero_hypothesis(self):
        print()
        if self.pearson_pvalue >= 0.05:
            print("Нулевая гипотеза не отвергается")
        else:
            print("Распределение не нормально")

    def __str__(self):
        s = f'Коэффициент корреляции Пирсона = {self.pearson} \nPvalue = {self.pearson_pvalue}'
        self.check_zero_hypothesis()
        return s


class LinRegression:
    def __init__(self, pandas_df):
        pass

def is_all_ints(lst):
    return all(isinstance(i, nums.Number) for i in lst)


def make_noraml_lists(pandas_df: DataFrame, column_name_1: str, column_name_2: str):


    column_1 = list(pandas_df[column_name_1])
    column_2 = list(pandas_df[column_name_2])

    if len(column_1) != len(column_2):
        raise Exception("У колонок разная длинна")

    if not (is_all_ints(column_1) and (is_all_ints(column_2))):
        raise Exception("В колонках присутствует что-то помимо целых")

    normal_data = Normalizer(column_1, column_2)
    normal_data.make_normal()

    return normal_data.get_normal_lists()


# x = [-224, 600, -425, -293, 551, 836, 64, -990, -833, 409, 682, -105, -10, 412, 491, -449, -611, -64, -264, 293, 1000]
# y = [790, 39, 140, -296, 872, -191, 50, 248, -591, 858, 344, -801, 788, 925, -439, -311, -786, 611, -423, 179, 1]


df = pd.read_csv(FILE_PATH, sep=',')

normal_x, normal_y = make_noraml_lists(df, 'score', 'gdp')

