// 10-301 programming hw 2
// Name: Minghan Gang
// Andrew ID: mgang
// Date: 09/18/2020

#include <iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<tgmath.h>

using namespace std;

string input_train, input_test, output_train, output_test, output_metric;
int max_depth, cnt_attrbt; // note that cnt_attrbt is the number including result
double train_cnt, train_diff, test_cnt, test_diff;
string line, buf;
string name_1, name_2;

// tree node structure
struct node {
    struct node *left_node;
    struct node *right_node;
    string left_attrbt;
    string right_attrbt;
    bool is_leaf;
    string result; // either a 
};

struct data{
    vector<string> data_content;
    bool valid;
};

vector<string> attributes;
vector<struct data> dataset, testset;

bool sameClass(vector<struct data> dataset) {
    string result_a = "";
    int num_line = dataset.size();
    int num_attrbt = dataset[0].data_content.size() - 1;
    for(int i = 0; i < num_line; i++) {
        if(result_a == "") {
            result_a = dataset[i].data_content[num_attrbt];
        } else if (result_a.compare(dataset[i].data_content[num_attrbt])) {
            return false;
        }
    }
    return true;
}

double calculateMutualInfo(double cnt_aa, double cnt_ab, double cnt_ba, double cnt_bb) {

    double total_res_a = cnt_aa + cnt_ba;
    double total_res_b = cnt_ab + cnt_bb;
    double total_res = total_res_a + total_res_b;

    // calculating the entropy
    double entropy = (total_res_a / total_res) * (total_res_a == 0? 0 : log2(total_res_a / total_res));
    entropy += (total_res_b / total_res) * (total_res_b == 0? 0 : log2(total_res_b / total_res));
    entropy = -entropy;

    double cnt_a = cnt_aa + cnt_ab;
    double cnt_b = cnt_ba + cnt_bb;
    double total_attrbt = cnt_a + cnt_b;

    double YgX_a = (cnt_aa / cnt_a) * (cnt_aa == 0? 0 : log2(cnt_aa / cnt_a)) + (cnt_ab / cnt_a) * (cnt_ab == 0? 0 : log2(cnt_ab / cnt_a));
    YgX_a = -YgX_a;
    YgX_a *= (cnt_a / total_attrbt);

    double YgX_b = (cnt_ba / cnt_b) * (cnt_ba == 0? 0 : log2(cnt_ba / cnt_b)) + (cnt_bb / cnt_b) * (cnt_bb == 0? 0 : log2(cnt_bb / cnt_b));
    YgX_b = -YgX_b;
    YgX_b *= (cnt_b / total_attrbt);

    double YgX = YgX_a + YgX_b;

    return entropy - YgX;
}

int bestAttribute(vector<struct data> dataset, vector<string> attributes) {
    int num_line = dataset.size();
    int num_attrbt = dataset[0].data_content.size() - 1; // we minus 1, because last one is result
    int best_attrbt = -1;
    double max_mutual_info = -1;

    for(int i = 0; i < num_attrbt; i++) {

        string val_a, val_b, res_a, res_b;
        val_a = val_b = res_a = res_b = "";
        double cnt_aa, cnt_ab, cnt_ba, cnt_bb;
        cnt_aa = cnt_ab = cnt_ba = cnt_bb = 0; // first one being attrbt

        double mutual_info = -1;

        for(int j = 0; j < num_line; j++) {
            string val = dataset[j].data_content[i]; // attribute value
            string res = dataset[j].data_content[num_attrbt]; // result value

            // initializing attribute value
            if(val_a == "") {
                val_a = val;
            } else if(val_b == "" && val.compare(val_a)) {
                val_b = val;
            }
            // initializing result value
            if(res_a == "") {
                res_a = res;
            } else if(res_b == "" &&  res.compare(res_a)) {
                res_b = res;
            }

            if(!val.compare(val_a) && !res.compare(res_a)) {
                cnt_aa++;
            } else if(!val.compare(val_a) && !res.compare(res_b)) {
                cnt_ab++;
            } else if(!val.compare(val_b) && !res.compare(res_a)) {
                cnt_ba++;
            } else if(!val.compare(val_b) && !res.compare(res_b)) {
                cnt_bb++;
            }  
        }
        mutual_info = calculateMutualInfo(cnt_aa, cnt_ab, cnt_ba, cnt_bb);
        if(mutual_info > max_mutual_info && mutual_info > 0) {
            max_mutual_info = mutual_info;
            best_attrbt = i;
        }
    }

    return best_attrbt; // if best_attrbt >= 0, then we can split
}

string majority_voter(vector<struct data> dataset) {
    
    int num_line = dataset.size();
     if(num_line == 0) {
        cout << "logical error!" << endl; 
    }
    int num_attrbt = dataset[0].data_content.size() - 1;
    string res_a, res_b;
    int cnt_a, cnt_b;
    res_a = res_b = "";
    cnt_a = cnt_b = 0;

    for(int i = 0; i < num_line; i++) {
        string res = dataset[i].data_content[num_attrbt];
        if(res_a == "") {
            res_a = res;
        } else if(res_b == "" && res.compare(res_a)) {
            res_b = res;
        }

        if(!res.compare(res_a)) {
            cnt_a++;
        } else if(!res.compare(res_b)) {
            cnt_b++;
        }
    }

    if(cnt_a > cnt_b) {
        return res_a;
    } else if(cnt_b > cnt_a) {
        return res_b;
    } else {
        return (res_a > res_b? res_a : res_b);
    }
}

struct node* buildTree(vector<struct data> dataset, vector<string> attributes, int depth, int best_att, string att) {
    // empty branch(which, in theory, is an impossible case)
    // because the mutual info would then be 0, but we only split when I > 0

    // for printing the tree
    for(int i = 0; i < depth; i++) {
        cout << "| ";
    }

    if(best_att != -1) {
        cout << attributes[best_att];
        cout << " = " << att << ": ";
    }
    int cnt_1, cnt_2;
    cnt_1 = cnt_2 = 0;
    int datasize = dataset.size();
    int res_idx = dataset[0].data_content.size();
    for(int i = 0; i < datasize; i++) {
        if(!name_1.compare(dataset[i].data_content[res_idx - 1])) {
            cnt_1++;
        } else if(!name_2.compare(dataset[i].data_content[res_idx - 1])) {
            cnt_2++;
        }
    }
    cout << "[" << cnt_1 << " " << name_1 << "/";
    cout << cnt_2 << " " << name_2 << "]" << endl;

    int num_line = dataset.size();

    if(num_line == 0) {
        return NULL;
    }

    int best_attrbt_idx = bestAttribute(dataset, attributes);

    node *res = new node;
    
    // majority vote 
    if(depth >= max_depth || attributes.size() - 1 <= 0 || sameClass(dataset) || best_attrbt_idx < 0) {
        res->left_node = NULL;
        res->right_node = NULL;
        res->left_attrbt = "";
        res->right_attrbt = "";
        res->is_leaf = true;
        res->result = majority_voter(dataset);
        return res;
    } else { // need to split
        vector<struct data> left, right;
        string attrbt_a, attrbt_b;
        attrbt_a = attrbt_b = "";

        for(int i = 0; i < num_line; i++) {
            string attrbt = dataset[i].data_content[best_attrbt_idx];
            if(attrbt_a == ""){
                attrbt_a = attrbt;
            } else if(attrbt_b == "" && attrbt_a.compare(attrbt)) {
                attrbt_b = attrbt;
            }
            
            if(!attrbt_a.compare(attrbt)) {
                left.push_back(dataset[i]);
            } else if(!attrbt_b.compare(attrbt)) {
                right.push_back(dataset[i]);
            }
        }

        res->left_attrbt = attrbt_a;
        res->right_attrbt = attrbt_b;
        res->left_node = buildTree(left, attributes, depth + 1, best_attrbt_idx, attrbt_a);
        res->right_node = buildTree(right, attributes, depth + 1, best_attrbt_idx, attrbt_b);
        stringstream ss;
        ss << best_attrbt_idx;
        res->result = ss.str();
        res->is_leaf = false;
    }
    return res;
}

string decide(struct node* Tree, vector<string> test) {

    if(Tree == NULL) {
        cout << "unexpected error: tree is NULL" << endl;
    }

    if(Tree->is_leaf) {
        return Tree->result;
    } else {
        int split_idx = stoi(Tree->result);
        if(test[split_idx] == Tree->left_attrbt) {
            return decide(Tree->left_node, test);
        } else if(test[split_idx] == Tree->right_attrbt) {
            return decide(Tree->right_node, test);
        }
    }
    return "";
}

void print_tree(struct node *Tree) {
    if(Tree != NULL) {
        if(Tree->left_node != NULL) {
            print_tree(Tree->left_node);
        }
        cout << Tree->result << endl;
        if(Tree->right_node != NULL) {
            print_tree(Tree->right_node);
        }
    }
}

int main(int argc, char **argv) {
    
    // making sure commandline input has 6 arguments
    try {
        if(argc != 7) {
            throw string("wrong number of commandline argument");
        }
    } catch (string e){
        cout << e << endl;
    }

//---- parsing commandline arguments ----//

    input_train = argv[1];
    input_test = argv[2];
    max_depth = atoi(argv[3]);
    output_train = argv[4];
    output_test = argv[5];
    output_metric = argv[6];
    name_1 = name_2 = "";

//---- reading the dataset from training input ----//

    ifstream ifs(input_train);

    getline(ifs, line);
    stringstream title(line);

    while(getline(title, buf, '\t')) {
        attributes.push_back(buf);
        cnt_attrbt++;
    }
    // closing the stringstream
    title.str("");

    while(getline(ifs, line)) {
        stringstream ss(line);
        vector<string> data;

        while(getline(ss, buf, '\t')) {
            data.push_back(buf);
        }

        if(name_1 == "") {
            name_1 = buf;
        } else if(name_2 == "" && buf.compare(name_1)) {
            name_2 = buf;
        }

        // false indicates that the data hasn't been split yet
        struct data instance;
        instance.data_content = data;
        instance.valid = false;

        dataset.push_back(instance);
    }

    ifs.close();
// ----building the tree---- //

    struct node *Tree = buildTree(dataset, attributes, 0, -1, "");

// ----metric output setup---- //

    ofstream metric_txt;
    metric_txt.open(output_metric);

// ----testing against training data---- //

    ifstream ifs2(input_train);
    ofstream train_label;
    train_label.open(output_train);

    getline(ifs2, line);

    while(getline(ifs2, line)) {
        stringstream ss(line);
        vector<string> test;


        while(getline(ss, buf, '\t')) {
            test.push_back(buf);
        }
        string train_res = decide(Tree, test);
        if(train_res.compare(dataset[train_cnt].data_content[cnt_attrbt - 1])) {
            train_diff++;
        }
        train_cnt++;
        train_label << train_res << endl;
    }
    metric_txt << "error(train): " << (train_diff / train_cnt) << endl;

// ---- testing agasin test data ---- //

    ifstream ifs3(input_test);
    ofstream test_label;
    test_label.open(output_test);

    getline(ifs3, line);

    while(getline(ifs3, line)) {
        stringstream ss(line);
        vector<string> test;

        while(getline(ss, buf, '\t')) {
            test.push_back(buf);
        }
        string test_res = decide(Tree, test);

        if(test_res.compare(test[cnt_attrbt - 1])) {
            test_diff++;
        }
        test_cnt++;
        test_label << test_res << endl;
    }
    metric_txt << "error(test): " << (test_diff / test_cnt) << endl;
    // remember to free the tree!
    
    return 0;
}