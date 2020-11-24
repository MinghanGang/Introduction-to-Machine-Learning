#include<iostream>
#include<fstream>
#include<sstream>
#include<map>
#include<string>

using namespace std;

string train_input, validation_input, test_input, dict_input, fomatted_train_output, formatted_validation_output, formatted_test_output, feature_flag;
string dict_line, train_line, validation_line, test_line;
string dictionary[40000];
string buf;
int ans[40000];
int cnt;
map<string, int> sdict;
map<string, int> s2dict;
map<string, int> s3dict;


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
    
    train_input = argv[1];
    validation_input = argv[2];
    test_input = argv[3];
    dict_input = argv[4];
    fomatted_train_output = argv[5];
    formatted_validation_output = argv[6];
    formatted_test_output = argv[7];
    feature_flag = argv[8];

    ifstream dict_s(dict_input);
    ifstream train_s(train_input);
    ifstream validation_s(validation_input);
    ifstream test_s(test_input);

    ofstream train_o;
    train_o.open(fomatted_train_output);
    ofstream validation_o;
    validation_o.open(formatted_validation_output);
    ofstream test_o;
    test_o.open(formatted_test_output);

    while(getline(dict_s, dict_line)) {
        stringstream dl(dict_line);
        getline(dl, buf, ' ');
        string word = buf;
        getline(dl, buf, ' ');
        int ind = stoi(buf);
        dictionary[ind] = word;
        cnt++;
    }

    while(getline(train_s, train_line)) {
        stringstream trainl(train_line);
        getline(trainl, buf, '\t');
        string y = buf;
        train_o << y << '\t';
        while(getline(trainl, buf, ' ')) {
            if(sdict[buf] != 0) {
                ans[sdict[buf] - 1]++;
            } else {
                for(int i = 0; i < cnt; i++) {
                    if(dictionary[i] == buf) {
                        ans[i]++;
                        sdict[buf] = i + 1;
                    }
                }
            }
            
        }
        for(int i = 0; i < cnt; i++) {
            if(feature_flag == "2") {
                if(ans[i] >= 1 && ans[i] < 4) {
                    train_o << to_string(i) << ":1\t";
                }
            } else if(feature_flag == "1") {
                if(ans[i] >= 1) {
                    train_o << to_string(i) << ":1\t";
                }
            }
        }
        train_o << endl;
        for(int i = 0; i < cnt; i++) {
            ans[i] = 0;
        }
    }

    while(getline(validation_s, validation_line)) {
        stringstream vl(validation_line);
        getline(vl, buf, '\t');
        string y = buf;
        validation_o << y << '\t';
        while(getline(vl, buf, ' ')) {
            if(s2dict[buf] != 0) {
                ans[s2dict[buf] - 1]++;
            } else {
                for(int i = 0; i < cnt; i++) {
                    if(dictionary[i] == buf) {
                        ans[i]++;
                        s2dict[buf] = i + 1;
                    }
                }
            }
        }
        for(int i = 0; i < cnt; i++) {
            if(feature_flag == "2") {
                if(ans[i] >= 1 && ans[i] < 4) {
                    validation_o << to_string(i) << ":1\t";
                }
            } else if(feature_flag == "1") {
                if(ans[i] >= 1) {
                    validation_o << to_string(i) << ":1\t";
                }
            }
        }
        validation_o << endl;
        for(int i = 0; i < cnt; i++) {
            ans[i] = 0;
        }
    }

    while(getline(test_s, test_line)) {
        stringstream testl(test_line);
        getline(testl, buf, '\t');
        string y = buf;
        test_o << y << '\t';
        while(getline(testl, buf, ' ')) {
            if(s3dict[buf] != 0) {
                ans[s3dict[buf] - 1]++;
            } else {
                for(int i = 0; i < cnt; i++) {
                    if(dictionary[i] == buf) {
                        ans[i]++;
                        s3dict[buf] = i + 1;
                    }
                }
            }
        }
        for(int i = 0; i < cnt; i++) {
            if(feature_flag == "2") {
                if(ans[i] >= 1 && ans[i] < 4) {
                    test_o << to_string(i) << ":1\t";
                }
            } else if(feature_flag == "1") {
                if(ans[i] >= 1) {
                    test_o << to_string(i) << ":1\t";
                }
            }
        }
        test_o << endl;
        for(int i = 0; i < cnt; i++) {
            ans[i] = 0;
        }
    }

    return 0;
}
