
// 10-301 programming hw 1
// Name: Minghan Gang
// Andrew ID: mgang
// Date: 09/10/2020

#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
using namespace std;

char *train_input, *test_input, *train_out, *test_out, *metrics_out;
int split_index;
string line;
string val1, val2;
string class_a, class_b;
int val1_a, val1_b, val2_a, val2_b;
string decision1, decision2;

int main(int argc, char **argv) {

    // not enough commandline argument provided
    if(argc < 6) {
        return 0;
    }

    // parsing commandline argument
    train_input = argv[1];
    test_input = argv[2];
    split_index = atoi(argv[3]);
    train_out = argv[4];
    test_out = argv[5];
    metrics_out = argv[6];

    val1 = val2 = class_a = class_b = "";

    double case_num = 0;
    ifstream ifs(train_input);

    // reading the title line
    getline(ifs, line);

    while(getline(ifs, line)) {
        case_num++;
        stringstream ss(line);

        string val, ans;

        string tmp;
        int cnt = 0;
        while(getline(ss, tmp, '\t')) {
            if(cnt == split_index) {
                val = tmp;
            }
            cnt++;
        }

        ans = tmp;

        // initializing the 2 values of the interested attribute
        if(val1 == "") {
            val1 = val;
        } else if(val2 == "" && val.compare(val1)) {
            val2 = val;
        }

        // initializing the two answers
        if(class_a == "") {
            class_a = ans;
        } else if(class_b == "") {
            class_b = ans;
        }
        
        if(!val.compare(val1)) {
            if(!ans.compare(class_a)) {
                val1_a++;
            } else if(!ans.compare(class_b)) {
                val1_b++;
            }
        } else if(!val.compare(val2)) {
            if(!ans.compare(class_a)) {
                val2_a++;
            } else if(!ans.compare(class_b)) {
                val2_b++;
            }
        }
    }

    // for train
    double error_1, error_2;
    double error_rate_train;

    if(val1_a > val1_b) {
        decision1 = class_a;
        error_1 = val1_b;
    } else {
        decision1 = class_b;
        error_1 = val1_a;
    }
    if(val2_a > val2_b) {
        decision2 = class_a;
        error_2 = val2_b;
    } else {
        decision2 = class_b;
        error_2 = val2_a;
    }

    error_rate_train = (error_1 + error_2) / case_num;

    ifs.close();

    ofstream train_label;
    train_label.open(train_out);

    ifstream ifs2(train_input);
    getline(ifs2, line);

    while(getline(ifs2, line)) {
        stringstream ss(line);
        string tmp;
        int cnt = 0;
        string val;
        while(getline(ss, tmp, '\t')) {
            if(cnt == split_index) {
                val = tmp;
            }
            cnt++;
        }
        if(!val.compare(val1)){
            train_label << decision1 << endl;
        } else if(!val.compare(val2)) {
            train_label << decision2 << endl;
        }

    }

    ofstream test_label;
    test_label.open(test_out);


    ifstream ifs3(test_input);
    getline(ifs3, line);
    
    double test_case_sum = 0;

    // for test
    double error_3, error_4;
    double error_rate_test;

    while(getline(ifs3, line)) {
        test_case_sum++;
        stringstream ss(line);
        string tmp;
        int cnt = 0;
        string val, actual_result;
        while(getline(ss, tmp, '\t')) {
            if(cnt == split_index) {
                val = tmp;
            }
            cnt++;
        }
        actual_result = tmp;
        if(!val.compare(val1)){
            test_label << decision1 << endl;
            if(actual_result.compare(decision1)) {
                error_3++;
            }
        } else if(!val.compare(val2)) {
            test_label << decision2 << endl;
            if(actual_result.compare(decision2)) {
                error_4++;
            }
        }
    }

    error_rate_test = (error_3 + error_4) / test_case_sum;

    ofstream metric;
    metric.open(metrics_out);
    metric << "error(train): " << error_rate_train << endl;
    metric << "error(test): " << error_rate_test << endl;


    return 0;
}