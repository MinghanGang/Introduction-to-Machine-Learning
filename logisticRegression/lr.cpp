#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include<vector>
#include<cmath>
#include<map>

using namespace std;

string train_in, valid_in, test_in, dict_in, train_out, test_out, metrics_out;
int num_epoch, cnt, train_cnt;
double train_error_cnt, test_error_cnt;
string buf, line;
double partial[40000], theta[40000];
vector<struct node> train_data, valid_data;
double alpha = 0.1;

struct node{
    double y;
    vector<int> val;
};

void update() {
    double dot_product;
    double exp_frac;
    double final_val;
    for(int i = 0; i <= cnt; i++) {
        partial[i] = 0;
    }
    for(int i = 0; i < train_data.size(); i++) {
        dot_product = 0;
        for(int j = 0; j < train_data[i].val.size(); j++)
            dot_product += theta[train_data[i].val[j]];
        double expp = exp(dot_product);
        exp_frac = expp / (1 + expp);
        final_val = train_data[i].y - exp_frac;

        for(int j = 0; j < train_data[i].val.size(); j++) {
            theta[train_data[i].val[j]] += alpha * final_val / train_data.size();
        }
    }
}

int main(int argc, char **argv) {
    try
    {
        if(argc != 9) {
            throw string("wrong number of commandline argument");
        }
    }
    catch(string e)
    {
        cout << e << endl;
    }

    train_in = argv[1];
    valid_in = argv[2];
    test_in = argv[3];
    dict_in = argv[4];
    train_out = argv[5];
    test_out = argv[6];
    metrics_out = argv[7];
    num_epoch = atoi(argv[8]);

    ifstream dict_s(dict_in);
    ifstream train_s(train_in);
    ifstream validation_s(valid_in);
    ifstream test_s(test_in);

    ofstream train_o;
    train_o.open(train_out);
    ofstream test_o;
    test_o.open(test_out);
    ofstream metrics_o;
    metrics_o.open(metrics_out);
    
    while(getline(dict_s, line)) {
        cnt++;
    }

    while(getline(train_s, line)) {
        stringstream trainl(line);
        getline(trainl, buf, '\t');
        struct node train_tmp;
        train_tmp.y = double(stoi(buf));

        while(getline(trainl, buf, '\t')) {
            int delim_pos = buf.find(':');
            train_tmp.val.push_back(stoi(buf.substr(0, delim_pos)));
        }
        train_tmp.val.push_back(cnt);
        train_data.push_back(train_tmp);
    }

    while(getline(validation_s, line)) {
        stringstream vl(line);
        getline(vl, buf, '\t');
        struct node valid_tmp;
        valid_tmp.y = double(stoi(buf));

        while(getline(vl, buf, '\t')) {
            int delim_pos = buf.find(':');
            valid_tmp.val.push_back(stoi(buf.substr(0, delim_pos)));
        }
        valid_tmp.val.push_back(cnt);
        valid_data.push_back(valid_tmp);
    }
    
    

    for(int i = 0; i < num_epoch; i++) {
        update();
        double tot = 0;
        for(int j = 0; j < train_data.size(); j++) {
            double target = train_data[j].y;
            double sum = 0;
            for(int m = 0; m < train_data[j].val.size(); m++) {
                sum += theta[train_data[j].val[m]];
            }
            tot += (-target) * sum + log(1 + exp(sum));
        }
        cout << tot / (double)train_data.size() << endl;
    }
    
    for(int i = 0; i < train_data.size(); i++) {
        double sum = 0;
        for(int j = 0; j < train_data[i].val.size(); j++) {
            sum += theta[train_data[i].val[j]];
        }
        double sigmoid = 1 / (1 + exp(-sum));
        if(sigmoid >= 0.5) {
            train_o << 1 << endl;
            if(train_data[i].y < 0.5) {
                train_error_cnt++;
            }
        } else {
            train_o << 0 << endl;
            if(train_data[i].y >= 0.5) {
                train_error_cnt++;
            }
        }
    }

    int test_cnt = 0;
    while(getline(test_s, line)) {
        test_cnt++;
        stringstream testl(line);
        getline(testl, buf, '\t');
        double target = double(stoi(buf));
        vector<int> test_line;
        double sum = 0;
        while(getline(testl, buf, '\t')) {
            int delim_pos = buf.find(':');
            sum += theta[stoi(buf.substr(0, delim_pos))];
        }
        sum += theta[cnt];
        double sigmoid = 1 / (1 + exp(-sum));
        if(sigmoid >= 0.5) {
            test_o << 1 << endl;
            if(target < 0.5) {
                test_error_cnt++;
            }
        } else {
            test_o << 0 << endl;
            if(target >= 0.5) {
                test_error_cnt++;
            }
        }
    }

    metrics_o << "error(train): " << train_error_cnt / (double)train_data.size() << endl;
    metrics_o << "error(test): " << test_error_cnt / (double)test_cnt;
    

    return 0;
}