import cmath
import csv
import itertools
import pickle
import typing
import unicodedata
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

#from app.interface import Region


def get_vertical_profile(grayscale: np.ndarray):
    """Вычисление вертикального профиля (подсчёт значений в горизонтальных рядах пикселей)"""
    assert grayscale.ndim == 2, "Должно быть двумерное изображение"
    if grayscale.max(initial=0) > 1:
        grayscale = grayscale / 255
    return grayscale.sum(axis=1).astype(np.uint8)


def get_horizontal_profile(grayscale: np.ndarray):
    """Вычисление горизонтального профиля (подсчёт значений в вертикальных столбцах пикселей)"""
    assert grayscale.ndim == 2, "Должно быть двумерное изображение"
    if grayscale.max(initial=0) > 1:
        grayscale = grayscale / 255
    return grayscale.sum(axis=0).astype(np.uint8)


def remove_black_padding(binary: np.ndarray):
    """Удаление у границ изображения рядов пикселей, содержащих только чёрные пиксели"""
    top = None
    bottom = None
    for y, value in enumerate(binary):
        if (value > 0).any():
            bottom = y + 1
            if top is None:
                top = y
    if top is None:
        top = 0
    left = None
    right = None
    for x, value in enumerate(binary.swapaxes(0, 1)):
        if (value > 0).any():
            right = x + 1
            if left is None:
                left = x
    if left is None:
        left = 0
    return binary[top:bottom, left:right]


def HuMoments(moments: Dict[str, float]) -> Sequence[float]:
    """8 Hu moments (standard 7 Hu moments plus one more)

    by the way, hu3 == (hu5**2 + hu7**2) / hu4**3

    reference: https://en.wikipedia.org/wiki/Image_moment, section "Rotation invariants"
    """
    nu02 = moments["nu02"]
    nu11 = moments["nu11"]
    nu20 = moments["nu20"]
    nu03 = moments["nu03"]
    nu12 = moments["nu12"]
    nu21 = moments["nu21"]
    nu30 = moments["nu30"]
    hu8 = nu11 * ((nu30 + nu12) ** 2 - (nu03 + nu21) ** 2) - (nu20 - nu02) * (nu30 + nu12) * (nu03 + nu21)
    standard_hu_moments: Sequence[float] = cv2.HuMoments(moments).reshape((7,))
    return tuple(standard_hu_moments) + (hu8,)


def calculate_diameter(contour) -> float:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    return radius


def shrink_size(new_size, array, aggregate):
    length = len(array)
    right = 0
    result = np.zeros(new_size)
    for i in range(new_size):
        left = right
        right = (length * (i + 1) + new_size - 1) // new_size
        result[i] = aggregate(array[left:right])
    return result


def calculate_fourier_descriptors(contour: np.ndarray, rotation=0., shift=(0., 0.), scale_factor=1.):
    """Вычисление дескрипторов Фурье

    reference: Digital image processing. Gonzalez R., Woods R.
      Цифровая обработка изображений Гонсалес Р., Вудс р.
    """
    if contour is None:
        return np.array((), dtype=np.complex128)
    s = np.zeros(len(contour), dtype=np.complex_)
    for i, [[x, y]] in enumerate(contour):
        s[i] = x + y * 1j
    s = s + shift[0] + shift[1] * 1j  # shift
    s = s * scale_factor  # scale
    s = s * cmath.exp(1j * rotation)  # rotate
    return np.fft.fft(s)


def safe_mean(arr, default=np.nan):
    if len(arr) == 0:
        return default
    return np.mean(arr)


def _safe_mean(arr):
    return safe_mean(arr, default=0)


def _shrink_profile(new_size: int, profile, prefix: str):
    profile = shrink_size(new_size=new_size, array=profile, aggregate=_safe_mean)
    return {f'{prefix}{i}_{new_size}': v for i, v in enumerate(profile, 1)}


def _shrink_fourier(n: int, fourier_descriptors: Sequence[complex], prefix: str):
    fourier_descriptors = fourier_descriptors[:n]
    if len(fourier_descriptors) < n:
        fourier_descriptors = np.pad(fourier_descriptors, (0, n - len(fourier_descriptors)))
    result = {}
    for i, f in enumerate(fourier_descriptors, 1):
        f: complex
        result[f'{prefix}r{i}'] = f.real
        result[f'{prefix}i{i}'] = f.imag
    return result


def calculate_orientation_angle(cov_matrix: np.ndarray) -> float:
    if cov_matrix[0, 0] == cov_matrix[1, 1]:
        return 0
    return np.arctan(2 * cov_matrix[0, 1] / (cov_matrix[0, 0] - cov_matrix[1, 1])) / 2


def calculate_eccentricity(cov_matrix: np.ndarray) -> float:
    """

    reference: https://en.wikipedia.org/wiki/Image_moment, section "Central moments"
    """
    D = 4 * (cov_matrix[0, 1] ** 2) + (cov_matrix[0, 0] - cov_matrix[1, 1]) ** 2
    # eigenvalues of covariance matrix
    l1 = (cov_matrix[0, 0] + cov_matrix[1, 1] + np.sqrt(D)) / 2
    l2 = (cov_matrix[0, 0] + cov_matrix[1, 1] - np.sqrt(D)) / 2
    if l2 > l1:
        l1, l2 = l2, l1
    if l1 == 0:
        return 1
    return np.sqrt(1 - l2 / l1)


def calculate_centroid(array: np.ndarray, is_binary_image: bool = False) -> Tuple[float, float]:
    M = cv2.moments(array, is_binary_image)
    if M['m00'] == 0:
        return 0, 0
    x = M['m10'] / M['m00']
    y = M['m01'] / M['m00']
    return x, y


def extract_features(binary: np.ndarray) -> Dict[str, float]:
    binary = remove_black_padding(binary)
    if binary.max() > 1:
        binary = binary // 255
    moments: Dict[str, float] = cv2.moments(binary, binaryImage=True)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cov_matrix = np.array([[moments["nu20"], moments["nu11"]], [moments["nu11"], moments["nu02"]]])
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours) or 1
    max_diameter = 0
    max_contour = None
    for contour in contours:
        diameter = calculate_diameter(contour)
        if diameter > max_diameter:
            max_contour = contour
            max_diameter = diameter
    h, w = binary.shape[:2]
    # centroid
    if moments["m00"] == 0:
        x = 0
        y = 0
    else:
        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]
    area = moments["m00"]
    angle = calculate_orientation_angle(cov_matrix)
    box_area = h * w
    vertical_profile = get_vertical_profile(binary) / w
    horizontal_profile = get_horizontal_profile(binary) / h
    normalized_fourier_descriptors = calculate_fourier_descriptors(
        max_contour,
        shift=(-moments["m10"], -moments["m01"]),
        rotation=-angle,
        scale_factor=1 / (max_diameter or 1),
    )
    other_features = {
        "S/H/W": area / box_area,
        "a": angle,
        "e": calculate_eccentricity(cov_matrix),
        "Rc": 4 * np.pi * area / perimeter ** 2,
        "X/W": x / w,
        "Y/H": y / h,
        "D/P": max_diameter / perimeter,
    }
    return {
        **moments,
        **other_features,
        **_shrink_profile(8, vertical_profile, prefix="Pv"),
        **_shrink_profile(8, horizontal_profile, prefix="Ph"),
        **_shrink_fourier(n=8, fourier_descriptors=normalized_fourier_descriptors, prefix="Fn"),
    }


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def index2name(clf, prediction_groups: Iterator[Sequence[int]]) -> Iterator[Sequence[str]]:
    for predictions in prediction_groups:
        yield [clf.classes_[p] for p in predictions]


def choose_features():
    chosen_columns = []
    chosen_columns += ['Rc', 'D/P', 'X/W', 'Y/H', 'e']
    chosen_columns += [f'Fni{i}' for i in range(1, 5)]
    chosen_columns += [f'Fnr{i}' for i in range(1, 5)]
    chosen_columns += [f'Pv{i}_8' for i in range(1, 9)]
    chosen_columns += [f'Ph{i}_8' for i in range(1, 9)]
    chosen_columns += ['nu11', 'nu12', 'nu21']
    return chosen_columns


def get_target_classes_and_names(y):
    """Use it to plot confusion matrix with readable target names"""
    target_classes = pd.unique(y)
    # target_names = pd.unique(y)
    target_names = []
    for class_ in target_classes:
        try:
            name = unicodedata.lookup(class_)
        except KeyError:
            name = class_
        target_names.append(name)
    pairs = [(name, class_) for name, class_ in zip(target_names, target_classes)]
    pairs.sort()
    target_classes = [class_ for _, class_ in pairs]
    target_names = [name for name, _ in pairs]
    return target_classes, target_names


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """given a sklearn confusion matrix (cm), make a nice plot

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    :param cm: confusion matrix from sklearn.metrics.confusion_matrix
    :param target_names: given classification classes such as [0, 1, 2] the class names,
                         for example: ['high', 'medium', 'low']
    :param title: the text to display at the top of the matrix
    :param cmap: the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    :param normalize: If False, plot the raw numbers
                      If True, plot the proportions
    """
    import matplotlib.pyplot as plt
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    mis_class = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(20, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; mis_class={:0.4f}'.format(accuracy, mis_class))
    plt.show()


def iter_features_from_train_samples(dataset_directory: typing.Union[Path, str]):
    dataset_directory = Path(dataset_directory)
    for symbol_directory in sorted(dataset_directory.iterdir()):
        if not symbol_directory.is_dir():
            continue
        for file_path in symbol_directory.iterdir():
            image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            h, w = image.shape[:2]
            if h == 0 or w == 0:
                warnings.warn(f"bad symbol {file_path}")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            min_value = 16
            if h < min_value or w < min_value:
                # multiplier = np.ceil([min_value / h, min_value / w]).max()
                multiplier = 8
                gray = cv2.resize(gray, None, None, multiplier, multiplier, cv2.INTER_LINEAR)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            features: typing.Dict[str, typing.Union[float, str]] = extract_features(binary)
            features['file_name'] = file_path.name
            features['letter'] = symbol_directory.name
            yield features


def save_features_to_csv(entries: typing.Iterable[typing.Dict[str, typing.Union[float, str]]],
                         output_path: typing.Union[str, Path]):
    with open(str(output_path), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        keys = None
        for features in entries:
            if keys is None:
                keys = list(features.keys())
                writer.writerow(keys)
            writer.writerow(features[key] for key in keys)


def prepare():
    dataset_directory = Path(__file__).parent / 'samples'
    output_path = Path(__file__).parent / 'features.csv'
    entries = iter_features_from_train_samples(dataset_directory)
    save_features_to_csv(entries, output_path)


def prepare_dataset(df: pd.DataFrame):
    if 'letter' in df:
        # В классификации участвуют только обычные буквы
        df = df[df['letter'].str.startswith('CYRILLIC SMALL LETTER')].copy()
        # объединение классов "иже" и "нашъ"
        i_class = 'CYRILLIC SMALL LETTER I'
        en_class = 'CYRILLIC SMALL LETTER EN'
        i_en_class = 'иже+нашъ'
        df['letter'] = df['letter'].replace({
            i_class: en_class,
            en_class: en_class,
        })
    return df


def train(df):
    chosen_columns = choose_features()
    X = df[chosen_columns].copy()
    y = df['letter']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = LogisticRegression(max_iter=100, solver='newton-cg', random_state=42)
    clf = make_pipeline(RobustScaler(), clf)
    print('Начало обучения')
    clf.fit(X_train, y_train)
    print('Окончание обучения')
    score = clf.score(X_test, y_test)
    print(f'Точность {score}, тренировочная {clf.score(X_train, y_train)}')
    # target_classes, target_names = get_target_classes_from_names(y)
    # cm = confusion_matrix(y_test, y_pred, labels=target_classes)
    # plot_confusion_matrix(cm, target_names=target_names, normalize=False)
    return clf


def _predict(clf, df: pd.DataFrame) -> Sequence[int]:
    chosen_features = clf.feature_names_in_
    X = df[chosen_features]
    y = clf.predict(X)
    return y


def _predict_n_best(clf, df: pd.DataFrame, n: int = 1) -> Sequence[Sequence[int]]:
    chosen_features = clf.feature_names_in_
    X = df[chosen_features]
    scores = clf.predict_proba(X)
    y = np.argsort(scores, axis=1)[:, ::-1][:, :n]
    # return y
    return y, scores


def predict_one(clf, binary: np.ndarray, n: int) -> Sequence[int]:
    record = extract_features(binary)
    df = pd.DataFrame.from_records([record])
    predictions = _predict_n_best(clf, df, n=n)
    return predictions[0]


def predict_many(clf, binaries: Iterator[np.ndarray], n: int) -> Sequence[Sequence[int]]:
    records = map(extract_features, binaries)
    df = pd.DataFrame.from_records(records)
    # predictions = _predict_n_best(clf, df, n=n)
    # return predictions
    predictions, score = _predict_n_best(clf, df, n=n)
    return predictions, score


#def iterate_crops(regions: List[Region], binary: np.ndarray):
#    for region in regions:
#        contour = np.array(region.polygon, dtype=np.int32)
#        contour = np.expand_dims(contour, axis=1)
#        yield contour_crop(binary, contour)


def contour_crop(binary: np.ndarray, contour: np.ndarray) -> np.ndarray:
    left, top, width, height = cv2.boundingRect(contour)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [contour], 255, cv2.LINE_8, 0, (-left, -top))
    chunk = 255 - binary[top: top + height, left: left + width]
    result = mask & chunk
    if result.sum() == 0:
        mask = 255 - mask
        result = mask & chunk
    return result


def rect_crop(binary: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    left, top, width, height = rect
    return binary[top:top + height, left: left + width]

