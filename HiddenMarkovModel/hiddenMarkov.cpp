#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include<Eigen/Dense>
#include<map>
#include<vector>

using namespace std;
using namespace Eigen;

string valid_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans, predicted_file, metric_file;
string line, buf;
double word_num, tag_num;
double error_num;
map<string, int> dict_word;
map<string, int> dict_tag;
map<int, string> dict_tag_rev;
map<int, string> dict_word_rev;
double ssum;
double tot_valid;
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

    valid_input = argv[1];
    index_to_word = argv[2];
    index_to_tag = argv[3];
    hmmprior = argv[4];
    hmmemit = argv[5];
    hmmtrans = argv[6];
    predicted_file = argv[7];
    metric_file = argv[8];

    //----word index extraction----
    ifstream idx_to_w_s(index_to_word);

    int index_cnt = 1;
    while(getline(idx_to_w_s, line)) {

        dict_word.insert(pair<string, int> (line, index_cnt));
        dict_word_rev.insert(pair<int, string> (index_cnt, line));
        word_num++;
    }
cout << "---------------"<< endl;
    //----tag index extraction----
    ifstream idx_to_t_s(index_to_tag);

    index_cnt = 1;
    while(getline(idx_to_t_s, line)) {
        dict_tag[line] = index_cnt;
        dict_tag_rev[index_cnt++] = line;
        tag_num++;
    }

    //----initializing PI----
    ifstream prior_s(hmmprior);

    MatrixXf PI((int)tag_num, 1);
    int pi_cnt = 0;
    while(getline(prior_s, line)) {
        double entry = stod(line);
        PI(pi_cnt, 0) = entry;
        pi_cnt++;
    }
    
    //----initializing A----
    ifstream trans_s(hmmtrans);

    int row_idx, col_idx;
    row_idx = col_idx = 0;

    MatrixXf A((int)tag_num, (int)tag_num);
    while(getline(trans_s, line)) {

        stringstream trans_line(line);
        col_idx = 0;
        while(getline(trans_line, buf, ' ')) {
            double entry = stod(buf);
            A(row_idx, col_idx) = entry;
            col_idx++;
        }
        row_idx++;
    }

    //----initializing B----
    ifstream emmit_s(hmmemit);
    row_idx = col_idx = 0;

    MatrixXf B((int)tag_num, (int)word_num);
    while(getline(emmit_s, line)) {
        stringstream emmit_line(line);
        col_idx = 0;
        while(getline(emmit_line, buf, ' ')) {
            double entry = stod(buf);
            B(row_idx, col_idx) = entry;
            col_idx++;
        }
        row_idx++;
    }

    //----reading valid_inputs----
    ifstream valid_s(valid_input);
    ofstream predict_o(predicted_file);
    ofstream metric_o(metric_file);

    int line_num = 0;
    while(getline(valid_s, line)) {
        stringstream valid_line(line);//alpha
        stringstream valid_line2(line);//beta
        

        vector<string> back_word;
        vector<string> back_tag;

        int word_cnt = 0;

        while(getline(valid_line, buf, ' ')) {
            int pos = buf.find('_');
            string word = buf.substr(0, pos);
            string tag = buf.substr(pos + 1);
            back_word.push_back(word);
            back_tag.push_back(tag);
            word_cnt++;
        }
        
        //----alpha calculation----
        MatrixXf alpha((int)word_cnt, (int)tag_num);
        for(int i = 0; i < word_cnt; i++){
            string word = back_word[i];
            cout << word <<" : " << dict_word[word] << endl;
            if(i == 0) {
                alpha.row(0) = (PI.cwiseProduct(B.col(dict_word[word] - 1))).transpose();
            } else {
                alpha.row(i) = (B.col(dict_word[word] - 1).cwiseProduct(A.transpose() * ((alpha.row(i - 1)).transpose()))).transpose();
            }
        }
        //----beta calculation----
        MatrixXf beta((int)word_cnt, (int)tag_num);
        for(int i = word_cnt - 1; i >= 0; i--) {
            string word = back_word[i];
            if(i == word_cnt - 1) {
                
                MatrixXf tmp(1, (int)tag_num);
                for(int k = 0; k < tag_num; k++) {
                    tmp(0, k) = 1.0;
                }
                beta.row(word_cnt - 1) = tmp;
                
            } else {
                if(dict_word[back_word[i + 1]] - 1 < 0 || dict_word[back_word[i + 1]] - 1 >= word_num) {
                    cout << "sad3!";return 0;
                }
                beta.row(i) = (A * ((B.col(dict_word[back_word[i + 1]] - 1)).cwiseProduct((beta.row(i + 1)).transpose()))).transpose();
            }
            
        }
        MatrixXf result = alpha.cwiseProduct(beta);
        for(int i = 0; i < word_cnt; i++) {

            double max = -1;
            int pos = -1;
            for(int j = 0; j < tag_num; j++) {
                if(result(i, j) > max) {
                    pos = j;
                    max = result(i, j);
                }
            }
            if(i == word_cnt - 1) {
                predict_o << back_word[i] << "_" << dict_tag_rev[pos + 1];
            } else {
                predict_o << back_word[i] << "_" << dict_tag_rev[pos + 1] << " ";
            }
            
            if(back_tag[i] != dict_tag_rev[pos + 1]) {
                error_num++;
            }
        }
        predict_o << endl;
        line_num++;
        double line_sum = 0;
        for(int i = 0; i < tag_num; i++) {
            line_sum += alpha(word_cnt - 1, i);
        }
        ssum += log(line_sum);
        tot_valid += word_cnt;
    }
    metric_o << "Average Log-Likelihood: " << ssum / line_num << endl;
    metric_o << "Accuracy: " <<(tot_valid - error_num) / tot_valid;
    return 0;
}
