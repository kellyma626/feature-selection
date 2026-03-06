#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

// helper function for loading dataset from file
vector<vector<double>> load_data(string filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        double value;

        while (ss >> value)
            row.push_back(value);

        data.push_back(row);
    }
    return data;
}

// helper function for computing distance
double distance_calc(const vector<double>& a, const vector<double>& b, const vector<int>& features) {
    double dist = 0;

    for (int f : features) {
        double diff = a[f] - b[f];
        dist += diff * diff;
    }
    return dist;
}

// helper function for printing feature sets like {1,2,3}
void print_set(const vector<int>& set) {
    cout << "{";

    for (size_t i = 0; i < set.size(); i++) {
        cout << set[i];
        if (i != set.size() - 1)
            cout << ",";
    }
    cout << "}";
}

// performs leave-one-out cross validation using nearest neighbor
double leave_one_out_validation(const vector<vector<double>>& data, const vector<int>& features) {
    int correct = 0;
    int n = data.size();

    for (int i = 0; i < n; i++) {
        double nearest_distance = 1e9;
        double nearest_label = -1;

        // compare instance i to all other instances
        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            double dist = distance_calc(data[i], data[j], features);

            // update nearest neighbor
            if (dist < nearest_distance) {
                nearest_distance = dist;
                nearest_label = data[j][0];
            }
        }

        // check if classification is correct
        if (data[i][0] == nearest_label)
            correct++;
    }

    return (double)correct / n;
}

// Forward Selection Algorithm
// starts with an empty feature set and adds the feature that gives the highest accuracy at each level
void forward_selection(vector<vector<double>>& data, int num_features) {
    vector<int> current_set;
    vector<int> best_overall_set;
    double best_overall_accuracy = 0;

    cout << "\nBeginning search.\n";

    // each level adds one feature
    for (int level = 1; level <= num_features; level++) {
        int feature_to_add = -1;
        double best_accuracy_this_level = 0;

        // try adding every unused feature
        for (int feature = 1; feature <= num_features; feature++) {
            if (find(current_set.begin(), current_set.end(), feature) == current_set.end()) {
                vector<int> candidate = current_set;
                candidate.push_back(feature);

                double accuracy = leave_one_out_validation(data, candidate);

                cout << "Using feature(s) ";
                print_set(candidate);
                cout << " accuracy is " << accuracy * 100 << "%\n";

                // track best feature for this level
                if (accuracy > best_accuracy_this_level) {
                    best_accuracy_this_level = accuracy;
                    feature_to_add = feature;
                }
            }
        }

        // add the best feature found
        current_set.push_back(feature_to_add);

        cout << "Feature set ";
        print_set(current_set);
        cout << " was best, accuracy is " << best_accuracy_this_level * 100 << "%\n";

        // track the best subset overall
        if (best_accuracy_this_level > best_overall_accuracy) {
            best_overall_accuracy = best_accuracy_this_level;
            best_overall_set = current_set;
        }
    }

    cout << "\nFinished search!! The best feature subset is ";
    print_set(best_overall_set);
    cout << ", which has accuracy " << best_overall_accuracy * 100 << "%\n";
}

// Backward Elimination Algorithm
// starts with all features and removes one feature at a time, choosing the removal that gives the best accuracy
void backward_elimination(vector<vector<double>>& data, int num_features) {
    vector<int> current_set;

    // start with all features
    for (int i = 1; i <= num_features; i++)
        current_set.push_back(i);

    vector<int> best_overall_set = current_set;
    double best_overall_accuracy = leave_one_out_validation(data, current_set);

    cout << "\nBeginning search.\n";

    // continue until only one feature remains
    while (current_set.size() > 1) {
        int feature_to_remove = -1;
        double best_accuracy_this_level = 0;

        // try removing each feature
        for (int feature : current_set) {
            vector<int> candidate = current_set;

            candidate.erase(remove(candidate.begin(), candidate.end(), feature), candidate.end());

            double accuracy = leave_one_out_validation(data, candidate);

            cout << "Using feature(s) ";
            print_set(candidate);
            cout << " accuracy is " << accuracy * 100 << "%\n";

            if (accuracy > best_accuracy_this_level) {
                best_accuracy_this_level = accuracy;
                feature_to_remove = feature;
            }
        }

        // remove the feature that produced the best accuracy
        current_set.erase(remove(current_set.begin(), current_set.end(), feature_to_remove), current_set.end());

        cout << "Feature set ";
        print_set(current_set);
        cout << " was best, accuracy is " << best_accuracy_this_level * 100 << "%\n";

        // track best subset found overall
        if (best_accuracy_this_level > best_overall_accuracy) {
            best_overall_accuracy = best_accuracy_this_level;
            best_overall_set = current_set;
        }
    }

    cout << "\nFinished search!! The best feature subset is ";
    print_set(best_overall_set);
    cout << ", which has accuracy " << best_overall_accuracy * 100 << "%\n";
}

// Driver Code
int main() {
    cout << fixed << setprecision(1);
    cout << "Welcome to Bertie Wooster's Feature Selection Algorithm.\n\n";

    string filename;

    cout << "Type the name of the file to test: ";
    cin >> filename;

    cout << "Type the number of the algorithm you want to run.\n";
    cout << "1) Forward Selection\n";
    cout << "2) Backward Elimination\n";

    int choice;
    cin >> choice;

    vector<vector<double>> data = load_data(filename);

    int num_features = data[0].size() - 1;
    int num_instances = data.size();

    cout << "\nThis dataset has "
         << num_features
         << " features (not including the class attribute), with "
         << num_instances
         << " instances.\n\n";

    vector<int> all_features;

    for (int i = 1; i <= num_features; i++)
        all_features.push_back(i);

    double accuracy = leave_one_out_validation(data, all_features);

    cout << "Running nearest neighbor with all "
         << num_features
         << " features, using \"leaving-one-out\" evaluation, I get an accuracy of "
         << accuracy * 100 << "%\n\n";

    if (choice == 1)
        forward_selection(data, num_features);
    else
        backward_elimination(data, num_features);

    return 0;
}