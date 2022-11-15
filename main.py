from scipy import stats
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import numbers as nums
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel as GLM

import os

FILE_PATH = 'file.txt'


def count_avg(numbers):
    return float(sum(numbers)) / len(numbers)


def count_skew_crit(n):
    ans = 3 * (6 * (n - 1) / ((n + 1) * (n + 3))) ** 0.5
    return ans


def count_kurtosis_crit(n):
    ans = 5 * (24 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))) ** 0.5
    return ans


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

        self.result = ''

    def check_zero_hypothesis(self):
        if self.pearson_pvalue >= 0.05:
            return "Нулевая гипотеза не отвергается"
        else:
            return "Распределение не нормально"

    def total(self):
        self.result += f"\n{self.check_zero_hypothesis()}\n"
        self.result += f'Коэффициент корреляции Пирсона = {self.pearson} \nPvalue = {self.pearson_pvalue}'


class LinRegression:
    def __init__(self, df: DataFrame, target_col: str, certain_cols: list = None, method: str = "OLS"):
        self.df = df
        self.target_col = target_col
        self.certain_cols = certain_cols
        self.method = method

        self.x = df.loc[:, df.columns != target_col]
        self.y = df.loc[:, target_col]

        self.result = None

        if not (self.certain_cols is None):
            if not (all(isinstance(i, str) for i in self.certain_cols)):
                raise Exception(' Выбранные столбцы должны быть строкой\n')
            self.x = df.loc[:, certain_cols]

    def count_regression(self):
        x = sm.add_constant(self.x)
        model = sm.OLS(self.y, x).fit()

        if self.method == "GLM":
            model = self.Maximum_Like(self.y, x).fit()

        elif self.method == "Logit":
            model = sm.Logit(self.y, x).fit(disp=0)

        elif self.method == "Probit":
            probit_mod = sm.Probit(self.y, x)
            probit_res = probit_mod.fit()
            model = probit_res.get_margeff()

        self.result = model.summary()
        #print(model.summary())

    class Maximum_Like(GLM):
        def loglike(self, params):
            exog = self.exog
            endog = self.endog

            q = 2 * endog - 1
            return stats.norm.logcdf(q * np.dot(exog, params)).sum()


class Inteface:
    def __init__(self):
        self.path = None
        self.sep = ','
        self.dataframe: DataFrame

    def greeting(self):
        path_to_file = r"C:\Users\igser\PycharmProjects\project_avito_parser\file.txt"
        #input("Здравствуйте, введите путь к файлу: ")
        sep = ''
        #input("И разделитель(по умолчанию ','): ")
        if sep != "":
            self.sep = sep

        while not os.path.exists(path_to_file):
            path_to_file = input("Файл по введенному пути не существует, попробуйте еще: ")

        self.path = path_to_file

        self.dataframe = pd.read_csv(self.path, sep=self.sep)

    def main_menu(self):

        print("Вот содержимое файла:")
        print(self.dataframe.head())

        ans = input("\nВыберите действие: 1) Перейти к расчетам   2) Вывести столбец:\n").strip()

        if ans not in ["1", "2"]:
            self.print_err()
            self.main_menu()

        if ans == "1":
            self.calculation_variant()

        else:
            pass

    def print_column(self):
        columns_dict = self.create_columns()
        self.print_columns(columns_dict)

        ans = input("\nВыберите колонку:\n")
        self.check_right_column_input(columns_dict, ans)

    def calculation_variant(self):
        ans = input("\n1) Корреляция Пирсона   2) Многофакторная линейная регрессия 3) Назад:\n").strip()

        if ans not in ["1", "2", "3"]:
            self.print_err()
            self.calculation_variant()

        if ans == "1":
            self.choose_column(variant=1)
        if ans == "2":
            self.choose_column(variant=2)
        if ans == "3":
            self.main_menu()

    def choose_column(self, variant: int = 1):
        columns_dict = self.create_columns()
        raw_dict = columns_dict.copy()
        self.print_columns(columns_dict)

        ans_y = int(input("Введите столбец Y:\n"))
        self.check_right_column_input(columns_dict, ans_y)

        columns_dict[ans_y] += " - выбран как Y"

        y_name = raw_dict[ans_y]
        if variant == 1:
            ans_x = self.get_pearson_x_column_and_mark_it(columns_dict)
            x_name = raw_dict[ans_x]
            result = self.count_pearson(x_column=x_name, y_column=y_name)
            print(result)

        if variant == 2:
            ans_x = self.get_lmr_x_column_and_mark_them(columns_dict, ans_y)
            x_names = self.get_x_column_names(raw_dict, ans_x)
            result = self.count_lmr(x_column=x_names, y_column=y_name)
            print(result)

        self.print_columns(columns_dict)
        self.main_menu()

    def count_pearson(self, x_column, y_column):
        normal_x, normal_y = make_noraml_lists(self.dataframe, 'score', 'gdp')
        pearson = Pearson(normal_x, normal_y)
        pearson.total()
        return pearson.result

    def count_lmr(self, x_column, y_column, method="OLS"):
        my_regression = LinRegression(self.dataframe, y_column, x_column, method=method)
        my_regression.count_regression()
        return my_regression.result

    def get_pearson_x_column_and_mark_it(self, columns):
        ans_x = int(input("Введите столбец X:\n"))
        self.check_right_column_input(columns, ans_x)
        columns[ans_x] += " - выбран как X"
        return ans_x

    def get_lmr_x_column_and_mark_them(self, columns, ans_y):
        ans_x = input("\nВведите столбцы X через пробел (по умолчанию все кроме Y):\n")

        if ans_x == "":
            self.mark_all_x_columns(columns, ans_y)
            ans_x = list(columns).remove(ans_y)
            return ans_x
        else:
            self.check_right_column_input(columns, ans_x)

            ans_x = list(map(int, ans_x.split()))
            self.mark_define_x_columns(columns, ans_x)
            return ans_x

    @staticmethod
    def get_x_column_names(columns, x_columns):
        x_names = []
        for key, value in columns.items():
            if key in x_columns:
                x_names.append(value)
        return x_names


    @staticmethod
    def print_err():
        print("Такого варианта нет!")

    @staticmethod
    def print_columns(columns):
        print("-"*20)
        print("Вот все возможныйе столбцы: ")
        for key, value in columns.items():
            print("{0}: {1}".format(key, value))
        print("-" * 20)

    @staticmethod
    def mark_all_x_columns(columns, y_column):
        for key, value in columns.items():
            if key != y_column:
                columns[key] += " - выбран как X"

    @staticmethod
    def mark_define_x_columns(columns, x_columns):
        for key, value in columns.items():
            if key in x_columns:
                columns[key] += " - выбран как X"

    def check_right_column_input(self, columns, ans):
        a_set = set(columns)
        if type(ans) is str:
            b_set = set(map(int, ans.split()))
        else:
            b_set = {ans}

        if (a_set & b_set) != b_set:
            self.print_err()
            self.calculation_variant()

    def create_columns(self):
        columns = {}
        for i, col in enumerate(self.dataframe.columns):
            columns[i + 1] = col
        return columns

# x = [-224, 600, -425, -293, 551, 836, 64, -990, -833, 409, 682, -105, -10, 412, 491, -449, -611, -64, -264, 293, 1000]
# y = [790, 39, 140, -296, 872, -191, 50, 248, -591, 858, 344, -801, 788, 925, -439, -311, -786, 611, -423, 179, 1]


# df = pd.read_csv(FILE_PATH, sep=',')

# normal_x, normal_y = make_noraml_lists(df, 'score', 'gdp')
#
# my_regression = LinRegression(df, 'score', ['med_age', 'gdp', 'Population_2015'])
# my_regression.count_regression()

i = Inteface()
i.greeting()
i.main_menu()
