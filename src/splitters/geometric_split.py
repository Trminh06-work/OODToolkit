import os
from .base_splitter import BaseSplitter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import List


class BasicGeometricSplit(BaseSplitter):
    def __init__(self, seeds: List[int] = None, keep_size: bool = False, num_balls: int = 3):
        """
        This class involves several convex-based splitting and its reversed splitting:
            1. Single Hyperball
            2. Multiple Hyperballs
            3. Single Slab
            4. Semi-infinite Slab
            5. KMeans Hyperballs

        Parameters:
            seeds: List[int], None as default. For reproducibility
            keep_size: bool False as default. Set to True to keep the big-sized data, >1M samples
            num_balls: int, 3 as default. Specify number of hyperballs/KMeans clusters
        """
        super().__init__(seeds, keep_size)
        self.num_balls = num_balls


    """
        Supportive functions:
            _largest_distance(self, center)
            _data_within_ball(center, radius)
            _ball_selection(self, center)
            _random_sums(self, total_size, num)
            _construct_hyperplane(self, seed)
            _data_within_slab(self, normal_vec, b, delta)
            _data_one_side(self, normal_vec, b)
            _compute_proportion(self, X_test)
            _find_bounds(self, SEED)
    """
    def _largest_distance(self, center):
      """
          Returns the largest distance from the center to the existing points in X
      """
      return np.max(np.linalg.norm(self.X - center, axis=1))


    def _data_within_ball(self, center, radius):
      """
          Returns points that are within the Hyperball defined by "center" and "radius" parameters
      """
      return self.X[np.linalg.norm(self.X - center, axis=1) <= radius]


    def _ball_selection(self, center, size):
        """
            Returns the points revolving around "center" and occupies "size"% of the whole X
        """
        # CAUTION: the highest possible radius must be the distance from the center to the
        #          farthest point in the feature space, not the dataset's diameter
        lo = 0
        epsilon = 0.01
        high = self._largest_distance(center)

        while lo < high:
            mid = lo + (high - lo) / 2

            inclusive_data = self._data_within_ball(center, mid)
            if len(inclusive_data) / self.X.shape[0] < size:
              lo = mid + epsilon
            else:
              high = mid - epsilon

        return inclusive_data


    def _union(self, a, b):
        """
            This return a list of indices that are in either a or b.

            Assume:
              len(a) > len(b)

            Optimize time complexity using Small-to-Large Merging technique
        """
        if len(a) < len(b):
          b, a = a, b
        return a.union(b)


    def _random_sums(self, total_size, num):
        """
            Randomly generate a list of "num" elements, whose sum is equal to "total_size"
        """
        cuts = np.sort(np.random.uniform(high = total_size, size = num - 1))
        parts = np.diff(np.concatenate(([0], cuts, [total_size])))
        return parts


    def _construct_hyperplane(self, seed):
        """
            This function randomly choose a point in the given set of inputs,
            sample a normal vector from Gaussian distribution N(0, 1), with mean = 0 and std = 1
            then return the corresponding hyperplane.
        """

        np.random.seed(seed)
        point = self.X.iloc[np.random.randint(0, len(self.X))]
        normal_vec = np.random.normal(size = self.X.shape[1]) # normal_vec ~ N(0, 1)
        b = np.dot(normal_vec, point)

        return normal_vec, b, point


    def _data_within_slab(self, normal_vec, b, delta):
        """
            Returns the points in X whose distance from the Hyperplane, defined by "normal_vec", "b", is less than "delta"
        """
        # | (X.dot(normal_vector) - b / ||normal_vec||) | < delta
        return self.X[(np.abs(self.X @ normal_vec - b) / np.linalg.norm(normal_vec)) < delta]


    def _data_one_side(self, normal_vec, b):
        """
            Returns the points in X that lie on one side of the Hyperplane defined by "normal_vec", "b"
        """
        return self.X[self.X @ normal_vec - b < 0]


    def _compute_proportion(self, X_test):
        """
            Returns the proportion of sampled points over the whole set X
        """
        return len(X_test) / len(self.X)


    def _find_bounds(self, SEED):
        """
            Returns two hyperplanes, "lo" and "hi" such that one can sample less than TEST_SIZE and another can sample more than TEST_SIZE of the whole training set X,
            and an associated normal vector "normal_vec" which defines the direction of the chosen Hyperplanes.
        """
        np.random.seed(SEED)

        normal_vec, b, _ = self._construct_hyperplane(SEED)

        dist = np.abs(self.X @ normal_vec - b) / np.linalg.norm(normal_vec)
        mask = (self.X @ normal_vec - b) < 0

        lo_points = np.where(mask)[0]      # indices where mask == True
        hi_points = np.where(~mask)[0]     # indices where mask == False

        # hi_points are guaranteed to exist, but not lo_points.
        # Thus, we, alternatively, use the points that closest and farthest
        try:
            # indices of the farthest point
            lo = lo_points[np.argmax(dist[mask])] 
            hi = hi_points[np.argmax(dist[~mask])]

            # the farthest points
            lo, hi = self.X.iloc[lo], self.X.iloc[hi]
        except:
            lo = self.X[np.argmin(dist)]
            hi = self.X[np.argmax(dist)]

        # Find the upper and lower bound
        b = np.dot(normal_vec, hi)
        X_test = self._data_one_side(normal_vec, b)

        if self._compute_proportion(X_test) < self.test_size:
            lo, hi = hi, lo

        return normal_vec, lo, hi


    """
        Main functions:
            _single_hyperball(self)
            _reverse_single_hyperball(self)
            _multiple_hyperballs(self, num_balls)
            _reverse_multiple_hyperballs(self, num_balls)
            _single_slab(self)
            _reverse_single_slab(self)
            _semi_infinite_slab(self)
            _reverse_semi_infinite_slab(self)
            _kmeans_hyperballs(self, n_clusters)
            _reverse_kmeans_hyperballs(self, n_clusters)
    """

    def _single_hyperball(self):
        """
            Select the trainng points based on a random Hyperball and predefined TRAIN_SIZE
        """

        # Create directory if not exist
        output_dir = f"../data/splitted/{self.file_name}/Single_Hyperball"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, SEED in enumerate(self.SEEDS):
            np.random.seed(SEED)

            center = self.X.iloc[np.random.randint(0, len(self.X))]
            inclusive_data = self._ball_selection(center, self.train_size)

            index_list = inclusive_data.index.to_list()
            X_train = self.X.iloc[index_list]
            y_train = self.y.iloc[index_list]
            X_test = self.X.iloc[~self.X.index.isin(index_list)]
            y_test = self.y.iloc[~self.y.index.isin(index_list)]

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)

            # Save files using the idx
            path = os.path.join(output_dir, f"train_{idx}.parquet")
            df_train.to_parquet(path, index = False)
            path = os.path.join(output_dir, f"test_{idx}.parquet")
            df_test.to_parquet(path, index = False)


    def _reverse_single_hyperball(self):
        """
            Select the trainng points based on a random Hyperball and predefined TRAIN_SIZE
        """

        # Create directory if not exist
        output_dir = f"../data/splitted/{self.file_name}/Reverse_Single_Hyperball"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, SEED in enumerate(self.SEEDS):
            np.random.seed(SEED)

            center = self.X.iloc[np.random.randint(0, len(self.X))]
            inclusive_data = self._ball_selection(center, self.test_size)

            index_list = inclusive_data.index.to_list()
            X_train = self.X.iloc[~self.X.index.isin(index_list)]
            y_train = self.y.iloc[~self.y.index.isin(index_list)]
            X_test = self.X.iloc[index_list]
            y_test = self.y.iloc[index_list]

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)

            # This is to compare different ways of choosing train and test sets (Additional experiment)
            path = os.path.join(output_dir, f"train_{idx}.parquet")
            df_train.to_parquet(path, index = False)
            path = os.path.join(output_dir, f"test_{idx}.parquet")
            df_test.to_parquet(path, index = False)


    def _multiple_hyperballs(self):
        """
            Select the test points based on several random Hyperballs and predefined TEST_SIZE

            Parameters:
                num_balls:  the number of expected Hyperballs
        """

        # Create directory if not exist
        output_dir = f"../data/splitted/{self.file_name}/Multiple_Hyperballs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, SEED in enumerate(self.SEEDS):
            # Avoid accidental modification to the original dataset
            X = self.X.copy(deep = True)
            y = self.y.copy(deep = True)

            inclusive_data = set()

            np.random.seed(SEED)
            sub_train_sizes = self._random_sums(self.train_size, self.num_balls)

            for sub_train_size in sub_train_sizes:
                np.random.seed(SEED)

                center = X.sample(n = 1, random_state = SEED).to_numpy().squeeze()

                sub_inclusive_data = self._ball_selection(center, sub_train_size)
                sub_inclusive_data = sub_inclusive_data.index.to_list()
                sub_inclusive_data = set(sub_inclusive_data)

                # the Hyperballs possibly intersect each other, this lead to fewer data points than expected.
                # To address this, we omit the selected data points to guarantee the number of data obtained by each ball.
                sub_inclusive_data = sub_inclusive_data - inclusive_data
                X = X.drop(sub_inclusive_data)

                inclusive_data = self._union(inclusive_data, sub_inclusive_data)

            index_list = list(inclusive_data)
            X_train = self.X.iloc[index_list]
            y_train = self.y.iloc[index_list]
            X_test = self.X.iloc[~self.X.index.isin(index_list)]
            y_test = self.y.iloc[~self.y.index.isin(index_list)]

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)

            # Save files using the idx
            path = os.path.join(output_dir, f"train_{idx}.parquet")
            df_train.to_parquet(path, index = False)
            path = os.path.join(output_dir, f"test_{idx}.parquet")
            df_test.to_parquet(path, index = False)


    def _reverse_multiple_hyperballs(self, num_balls):
        # Create directory if not exist
        output_dir = f"../data/splitted/{self.file_name}/Reverse_Multiple_Hyperballs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, SEED in enumerate(self.SEEDS):
            # Avoid accidental modification to the original dataset
            X = self.X.copy(deep = True)
            y = self.y.copy(deep = True)

            inclusive_data = set()

            np.random.seed(SEED)
            sub_test_sizes = self._random_sums(self.test_size, num_balls)

            for sub_test_size in sub_test_sizes:
                np.random.seed(SEED)

                center = X.sample(n = 1, random_state = SEED).to_numpy().squeeze()

                sub_inclusive_data = self._ball_selection(center, sub_test_size)
                sub_inclusive_data = sub_inclusive_data.index.to_list()
                sub_inclusive_data = set(sub_inclusive_data)

                # the Hyperballs possibly intersect each other, this lead to fewer data points than expected.
                # To address this, we omit the selected data points to guarantee the number of data obtained by each ball.
                sub_inclusive_data = sub_inclusive_data - inclusive_data
                X = X.drop(sub_inclusive_data)

                inclusive_data = self._union(inclusive_data, sub_inclusive_data)

            index_list = list(inclusive_data)
            X_train = self.X.iloc[~self.X.index.isin(index_list)]
            y_train = self.y.iloc[~self.y.index.isin(index_list)]
            X_test = self.X.iloc[index_list]
            y_test = self.y.iloc[index_list]

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)

            # This is to compare different ways of choosing train and test sets (Additional experiment)
            path = os.path.join(output_dir, f"train_{idx}.parquet")
            df_train.to_parquet(path, index = False)
            path = os.path.join(output_dir, f"test_{idx}.parquet")
            df_test.to_parquet(path, index = False)


    def _single_slab(self):
        """
            Construct a slab and select data within the slab as training data based on TRAIN_SIZE, and those beyond the slab as testing data.
        """

        # Create directory if not exist
        output_dir = f"../data/splitted/{self.file_name}/Single_Slab"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, SEED in enumerate(self.SEEDS):
            normal_vec, b, point = self._construct_hyperplane(SEED)

            # binary search
            lo = 0
            epsilon = 0.01
            high = np.max(np.linalg.norm(self.X - point, axis = 1))

            while lo < high:
                delta = lo + (high - lo) / 2
                inclusive_data = self._data_within_slab(normal_vec, b, delta)
                if len(inclusive_data) / self.X.shape[0] < self.train_size:
                  lo = delta + epsilon
                else:
                  high = delta - epsilon

            X_train = self._data_within_slab(normal_vec, b, delta)
            y_train = self.y.iloc[X_train.index]
            X_test = self.X.iloc[~self.X.index.isin(X_train.index)]
            y_test = self.y.iloc[~self.y.index.isin(X_train.index)]

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)

            path = os.path.join(output_dir, f"train_{idx}.parquet")
            df_train.to_parquet(path, index = False)
            path = os.path.join(output_dir, f"test_{idx}.parquet")
            df_test.to_parquet(path, index = False)


    def _reverse_single_slab(self):
        # Create directory if not exist
        output_dir = f"../data/splitted/{self.file_name}/Reverse_Single_Slab"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, SEED in enumerate(self.SEEDS):
            normal_vec, b, point = self._construct_hyperplane(SEED)

            # binary search
            lo = 0
            epsilon = 0.01
            high = np.max(np.linalg.norm(self.X - point, axis = 1))

            while lo < high:
                delta = lo + (high - lo) / 2
                inclusive_data = self._data_within_slab(normal_vec, b, delta)
                if len(inclusive_data) / self.X.shape[0] < self.test_size:
                  lo = delta + epsilon
                else:
                  high = delta - epsilon

            X_test = self._data_within_slab(normal_vec, b, delta)
            y_test = self.y.iloc[X_test.index]
            X_train = self.X.iloc[~self.X.index.isin(X_test.index)]
            y_train = self.y.iloc[~self.y.index.isin(X_test.index)]

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)

            # This is to compare different ways of choosing train and test sets (Additional experiment)
            path = os.path.join(output_dir, f"train_{idx}.parquet")
            df_train.to_parquet(path, index = False)
            path = os.path.join(output_dir, f"test_{idx}.parquet")
            df_test.to_parquet(path, index = False)


    def _semi_infinite_slab(self):
        """
        Construct a slab and select data lies on one side of the slab to be the test data, based on TEST_SIZE
        """

        # Create directory if not exist
        output_dir = f"../data/splitted/{self.file_name}/Semi_Infinite_Slab"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, SEED in enumerate(self.SEEDS):
            np.random.seed(SEED)
            normal_vec, lo, high = self._find_bounds(SEED)
            epsilon = 0.01
            cur_proportion = 0

            # According to the lower and upperbounds, this function leverages "binary search" to find a point which lies in the hyperplane that
            # can split the dataset into two sets that satisfies the given size requirements. The termination condition is when the training set's size
            # belongs to [train_size, train_size + epsilon].
            while cur_proportion < self.train_size or cur_proportion > self.train_size + epsilon:
                point = lo + (high - lo) / 2

                b = np.dot(normal_vec, point)
                X_train = self._data_one_side(normal_vec, b)

                cur_proportion = self._compute_proportion(X_train)
                if cur_proportion < self.test_size:
                  lo = point + epsilon
                else:
                  high = point - epsilon

                if cur_proportion == self._compute_proportion(X_train):
                    break
                cur_proportion = self._compute_proportion(X_train)

            y_train = self.y.iloc[X_train.index]
            X_test = self.X.iloc[~self.X.index.isin(X_train.index)]
            y_test = self.y.iloc[~self.y.index.isin(X_train.index)]

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)

            # Save files using the idx
            path = os.path.join(output_dir, f"train_{idx}.parquet")
            df_train.to_parquet(path, index = False)
            path = os.path.join(output_dir, f"test_{idx}.parquet")
            df_test.to_parquet(path, index = False)


    def _reverse_semi_infinite_slab(self):
        # Create directory if not exist
        output_dir = f"../data/splitted/{self.file_name}/Reverse_Semi_Infinite_Slab"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, SEED in enumerate(self.SEEDS):
            np.random.seed(SEED)
            normal_vec, lo, high = self._find_bounds(SEED)
            epsilon = 0.01
            cur_proportion = 0

            # According to the lower and upperbounds, this function leverages "binary search" to find a point which lies in the hyperplane that
            # can split the dataset into two sets that satisfies the given size requirements. The termination condition is when the testing set's size
            # belongs to [test_size, test_size + epsilon].
            while cur_proportion < self.test_size or cur_proportion > self.test_size + epsilon:
                point = lo + (high - lo) / 2

                b = np.dot(normal_vec, point)
                X_test = self._data_one_side(normal_vec, b)

                cur_proportion = self._compute_proportion(X_test)
                if cur_proportion < self.test_size:
                  lo = point + epsilon
                else:
                  high = point - epsilon

                if cur_proportion == self._compute_proportion(X_test):
                    break
                cur_proportion = self._compute_proportion(X_test)

            y_test = self.y.iloc[X_test.index]
            X_train = self.X.iloc[~self.X.index.isin(X_test.index)]
            y_train = self.y.iloc[~self.y.index.isin(X_test.index)]

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)

            # This is to compare different ways of choosing train and test sets (Additional experiment)
            path = os.path.join(output_dir, f"train_{idx}.parquet")
            df_train.to_parquet(path, index = False)
            path = os.path.join(output_dir, f"test_{idx}.parquet")
            df_test.to_parquet(path, index = False)


    def _kmeans_hyperballs(self):
        """
            Select the test points based on several Hyperballs, whose centers are determined by KMeans Clustering technique and predefined TRAIN_SIZE

            Parameters:
                n_clusters:  the number of expected Hyperballs
        """
        # Create directory if not exist
        output_dir = f"../data/splitted/{self.file_name}/KMeans_Hyperballs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, SEED in enumerate(self.SEEDS):
            kmeans = KMeans(n_clusters = self.num_balls, random_state = SEED, n_init = 10)
            kmeans.fit(self.X)

            # centroids = kmeans.cluster_centers_
            labels = kmeans.labels_

            total = self.X.shape[0]
            clusters = [i for i in range(self.num_balls)]

            inclusive_data = set()

            while len(inclusive_data) / total < self.train_size and len(clusters):
                np.random.seed(SEED)
                cl = np.random.choice(clusters)
                labels_indices = np.where(labels == cl)[0]

                if (len(inclusive_data) + len(labels_indices)) / total < self.train_size:
                  inclusive_data = self._union(inclusive_data, set(labels_indices))
                else:
                  num = int(self.train_size * total - len(inclusive_data))
                  inclusive_data = self._union(inclusive_data, set(labels_indices[:num]))

                clusters.remove(cl)

            index_list = list(inclusive_data)

            X_train = self.X.iloc[index_list]
            y_train = self.y.iloc[index_list]
            X_test = self.X.iloc[~self.X.index.isin(index_list)]
            y_test = self.y.iloc[~self.y.index.isin(index_list)]

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)

            # Save files using the idx
            path = os.path.join(output_dir, f"train_{idx}.parquet")
            df_train.to_parquet(path, index = False)
            path = os.path.join(output_dir, f"test_{idx}.parquet")
            df_test.to_parquet(path, index = False)


    def _reverse_kmeans_hyperballs(self):
        """
            Select the test points based on several Hyperballs, whose centers are determined by KMeans Clustering technique and predefined TEST_SIZE
        """
        # Create directory if not exist
        output_dir = f"../data/splitted/{self.file_name}/KMeans_Hyperballs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, SEED in enumerate(self.SEEDS):
            kmeans = KMeans(n_clusters = self.num_balls, random_state = SEED)
            kmeans.fit(self.X)

            # centroids = kmeans.cluster_centers_
            labels = kmeans.labels_

            total = self.X.shape[0]
            clusters = [i for i in range(self.num_balls)]

            inclusive_data = set()

            while len(inclusive_data) / total < self.test_size and len(clusters):
                np.random.seed(SEED)
                cl = np.random.choice(clusters)
                labels_indices = np.where(labels == cl)[0]

                if (len(inclusive_data) + len(labels_indices)) / total < self.test_size:
                  inclusive_data = self._union(inclusive_data, set(labels_indices))
                else:
                  num = int(self.test_size * total - len(inclusive_data))
                  inclusive_data = self._union(inclusive_data, set(labels_indices[:num]))

                clusters.remove(cl)

            index_list = list(inclusive_data)
            X_train = self.X.iloc[~self.X.index.isin(index_list)]
            y_train = self.y.iloc[~self.y.index.isin(index_list)]
            X_test = self.X.iloc[index_list]
            y_test = self.y.iloc[index_list]

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)

            # Save files using the idx
            path = os.path.join(output_dir, f"train_{idx}.parquet")
            df_train.to_parquet(path, index = False)
            path = os.path.join(output_dir, f"test_{idx}.parquet")
            df_test.to_parquet(path, index = False)


    def split(self, file_name = None, df = None, test_size = None, include_reverse = False):
        super().split(file_name, df, test_size)
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
        self.test_size = test_size
        self.train_size = 1 - test_size

        self._single_hyperball()

        self._multiple_hyperballs()

        self._single_slab()

        self._semi_infinite_slab()

        self._kmeans_hyperballs()

        if include_reverse:
            self._reverse_single_hyperball()

            self._reverse_multiple_hyperballs()

            self._reverse_single_slab()

            self._reverse_semi_infinite_slab()

            self._reverse_kmeans_hyperballs()
