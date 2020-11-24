#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include<map>

using namespace std;

string train_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans;
string line, buf;
double first_tag_cnt[100000], tag_to_tag_cnt[10000][10000], tag_to_word_cnt[10000][10000];
double tag_num, word_num, line_num;
map<string, int> dict_word;
map<string, int> dict_tag;

int main(int argc, char **argv) {
    try
    {
        if(argc != 7) {
            throw string("wrong number of commandline argument");
        }
    }
    catch(string e)
    {
        cout << e << endl;
    }

    train_input = argv[1];
    index_to_word = argv[2];
    index_to_tag = argv[3];
    hmmprior = argv[4];
    hmmemit = argv[5];
    hmmtrans = argv[6];

//----word index extraction----
    ifstream idx_to_w_s(index_to_word);

    int index_cnt = 1;
    while(getline(idx_to_w_s, line)) {
        dict_word[line] = index_cnt++;
        word_num++;
    }

//----tag index extraction----
    ifstream idx_to_t_s(index_to_tag);

    index_cnt = 1;
    while(getline(idx_to_t_s, line)) {
        dict_tag[line] = index_cnt++;
        tag_num++;
    }

//----train data processing----
    ifstream train_in_s(train_input);

    while(getline(train_in_s, line)) {
        line_num++;
        stringstream trainline(line);
        int clause_cnt = 1;
        int prev_tag_idx = 0;
        while(getline(trainline, buf, ' ')) {

            //----counting the first tag for pi----
            if(clause_cnt == 1) {
                int pos = buf.find('_');
                string first_tag = buf.substr(pos + 1);
                string first_word = buf.substr(0, pos);
                tag_to_word_cnt[dict_tag[first_tag]][dict_word[first_word]]++;
                first_tag_cnt[dict_tag[first_tag]]++;
                if(prev_tag_idx != 0) {
                    tag_to_tag_cnt[prev_tag_idx][dict_tag[first_tag]]++;
                }
                prev_tag_idx = dict_tag[first_tag];    
            } else {
                //----couting transition of tags----
                int pos = buf.find('_');
                string tag = buf.substr(pos + 1);
                string word = buf.substr(0, pos);
                tag_to_word_cnt[dict_tag[tag]][dict_word[word]]++;
                if(prev_tag_idx != 0) {
                    tag_to_tag_cnt[prev_tag_idx][dict_tag[tag]]++;
                }
                prev_tag_idx = dict_tag[tag];
            }
            clause_cnt++;
        }
    }


    ofstream(hmmprior_o);
    hmmprior_o.open(hmmprior);
    for(int i = 1; i <= tag_num; i++) {
        first_tag_cnt[i] = (first_tag_cnt[i] + 1) / (tag_num + line_num);
        hmmprior_o << first_tag_cnt[i] << endl;
    }

    ofstream(hmmtrans_o);
    hmmtrans_o.open(hmmtrans);
    for(int i = 1; i <= tag_num; i++) {
        double line_sum = 0;
        for(int j = 1; j <= tag_num; j++) {
            line_sum += tag_to_tag_cnt[i][j];
        }
        for(int j = 1; j <= tag_num; j++) {
            tag_to_tag_cnt[i][j] = (tag_to_tag_cnt[i][j] + 1) / (tag_num + line_sum);
            hmmtrans_o << tag_to_tag_cnt[i][j] << " ";
        }
        hmmtrans_o << endl;
    }

    ofstream(hmmemit_o);
    hmmemit_o.open(hmmemit);
    for(int i = 1; i <= tag_num; i++) {
        double line_sum = 0;
        for(int j = 1; j <= word_num; j++) {
            line_sum += tag_to_word_cnt[i][j];
        }
        for(int j = 1; j <= word_num; j++) {
            tag_to_word_cnt[i][j] = (tag_to_word_cnt[i][j] + 1) / (line_sum + word_num);
            hmmemit_o << tag_to_word_cnt[i][j] << " ";
        }
        hmmemit_o << endl;
    }

    return 0;
}