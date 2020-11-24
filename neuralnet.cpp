#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include<vector>
#include<Eigen/Dense>
#include<cmath>

using namespace std;
using namespace Eigen;

string train_input, validation_input, train_output, validation_output, metrics_output;
string line, buf;
int num_epoch, hidden_units, init_flag;
float learning_rate;
float train_error, valid_error;

struct node{
    int label;
    MatrixXf X;
};

struct object{
    MatrixXf y_hat;
    float J;
    MatrixXf b;
    MatrixXf z;
    MatrixXf z_prime;
};

struct grad{
    MatrixXf grad_alpha;
    MatrixXf grad_beta;
};

vector<struct node> train_dataset;
vector<struct node> validation_dataset;

MatrixXf sigmoid_matrix(MatrixXf X) {
    int length = X.size();
    MatrixXf ret = MatrixXf::Constant(length + 1, 1, 0);
    for(int i = 1; i <= length; i++) {
        ret(i, 0) = 1 / (1 + exp(-(X(i - 1,0))));
    }
    ret(0, 0) = 1;
    return ret;
}

MatrixXf softmax(MatrixXf b) {
    int length = b.size();
    float sum = 0;
    for(int i = 0; i < length; i++) {
        sum += exp(b(i, 0));
    }
    MatrixXf ret = MatrixXf::Constant(length, 1, 0);
    for(int i = 0; i < length; i++) {
        ret(i, 0) = exp(b(i, 0)) / sum;
    }
    return ret;
}

float cross_entropy(MatrixXf y, MatrixXf y_hat) {
    float ret = 0;
    int length = y.size();
    for(int k = 0; k < length; k++) {
        ret += y(k,0) * log(y_hat(k, 0));
    }
    ret = -ret;
    return ret;
}

MatrixXf cross_entropy_backward(MatrixXf y_hat, MatrixXf y, float J) {
    int length = y_hat.size();
    MatrixXf ret = MatrixXf::Constant(1, length, 0);
    for(int k = 0; k < length; k++) {
        ret(0, k) = -y(k, 0)/y_hat(k, 0);
    }
    return ret;
}

MatrixXf softmax_backward(MatrixXf y_hat, MatrixXf y) {
    int length = y.size();
    MatrixXf ret = MatrixXf::Constant(1, length, 0);
    for(int i = 0; i < length; i++) {
        ret(0, i) = -y(i, 0) + y_hat(i, 0);
    }
    return ret;
}

MatrixXf linear_backward_beta(MatrixXf grad_b, MatrixXf z) {
    MatrixXf ret = grad_b.transpose() * z.transpose();
    return ret;
}

MatrixXf linear_backward_z(MatrixXf grad_b, MatrixXf beta) {
    MatrixXf ret = grad_b * beta.block(0, 1, 10, hidden_units);
    return ret;
}

MatrixXf sigmoid_backward(MatrixXf grad_z, MatrixXf z) {
    int length = grad_z.size();
    MatrixXf ret = MatrixXf::Constant(1, length, 0);
    for(int i = 0; i < length; i++) {
        ret(0, i) = grad_z(0, i) * z(i, 0) * (1 - z(i, 0));
    }
    return ret;
}

MatrixXf linear_forward(MatrixXf beta,MatrixXf z) {
    int length = z.size();
    MatrixXf ret = beta * z;
    return ret;
}

MatrixXf linear_backward(MatrixXf grad_a, MatrixXf X) {
    MatrixXf ret = (grad_a.transpose()) * (X.transpose());
    return ret;
}

struct object NNForward(MatrixXf X, MatrixXf y, MatrixXf alpha, MatrixXf beta) {
    struct object ret;
    MatrixXf a = alpha * X;
    MatrixXf z_prime = sigmoid_matrix(a);
    MatrixXf b = linear_forward(beta, z_prime);
    MatrixXf y_hat = softmax(b);
    float J = cross_entropy(y, y_hat);

    int length = z_prime.size();
    MatrixXf z = MatrixXf::Constant(length - 1, 1, 0.0);
    for(int i = 0; i < length - 1; i++) {
        z(i, 0) = z_prime(i + 1, 0);
    }
    ret.y_hat = y_hat;
    ret.J = J;
    ret.b = b;
    ret.z = z;
    ret.z_prime = z_prime;
    return ret;
}

struct grad NNBackward(struct object O, MatrixXf X, MatrixXf y, MatrixXf beta) {
    MatrixXf grad_y_hat = cross_entropy_backward(O.y_hat, y, O.J);
    MatrixXf grad_b = softmax_backward(O.y_hat, y);
    MatrixXf grad_beta = linear_backward_beta(grad_b, O.z_prime);
    MatrixXf grad_z = linear_backward_z(grad_b, beta);
    MatrixXf grad_a = sigmoid_backward(grad_z, O.z);
    MatrixXf grad_alpha = linear_backward(grad_a, X);

    struct grad ret;
    ret.grad_alpha = grad_alpha;
    ret.grad_beta = grad_beta;
    return ret;
}

MatrixXf y_hat_compute(MatrixXf beta, MatrixXf alpha, MatrixXf X) {
    MatrixXf sig = sigmoid_matrix(alpha * X);
    return softmax(beta * sig);
}

int main(int argc, char **argv) {
    try
    {
        if(argc != 10) {
            throw string("wrong number of commandline argument");
        }
    }
    catch(string e)
    {
        cout << e << endl;
    }

    // parsing commandline arguments
    train_input = argv[1];
    validation_input = argv[2];
    train_output = argv[3];
    validation_output = argv[4];
    metrics_output = argv[5];
    num_epoch = stoi(argv[6]);
    hidden_units = stoi(argv[7]);
    init_flag = stoi(argv[8]);
    learning_rate = stof(argv[9]);

    // initializing the parameter matrices
    MatrixXf alpha, beta;
    if(init_flag == 1) { // random initialization
        // alpha initialization
        alpha = MatrixXf::Random(hidden_units, 128 + 1); // 128 + 1
        alpha /= 10;
        alpha.col(0) = MatrixXf::Constant(hidden_units, 1, 0.0);
        // beta initialization
        beta = MatrixXf::Random(10, 1 + hidden_units);
        beta /= 10;
        beta.col(0) = MatrixXf::Constant(10, 1, 0.0);
    } else { // zero initialization
        // alpha initialization
        alpha = MatrixXf::Zero(hidden_units, 128 + 1); // 128 + 1
        // beta initialization
        beta = MatrixXf::Zero(10, 1 + hidden_units);
    }

    // reading the input files
    ifstream train_in_s(train_input);
    ifstream validation_in_s(validation_input);

    ofstream train_o;
    train_o.open(train_output);
    ofstream valid_o;
    valid_o.open(validation_output);
    ofstream metrics_o;
    metrics_o.open(metrics_output);
    ofstream metrics2_o;
    metrics2_o.open("valid_data");

    while(getline(train_in_s, line)) {
        stringstream trainline(line);
        struct node data;
        getline(trainline, buf, ',');
        data.label = stoi(buf);

        MatrixXf x;
        x = MatrixXf::Zero(128 + 1, 1);
        int cnt = 1;
        while(getline(trainline, buf, ',')) {
            x(cnt, 0) = stof(buf);
            cnt++;
        }
        x(0, 0) = 1;
        data.X = x;
        train_dataset.push_back(data);
    }

    while(getline(validation_in_s, line)) {
        stringstream validline(line);
        struct node valid;
        getline(validline, buf, ',');
        valid.label = stoi(buf);

        MatrixXf x;
        x = MatrixXf::Zero(128 + 1, 1);
        int cnt = 1;
        while(getline(validline, buf, ',')) {
            x(cnt, 0) = stof(buf);
            cnt++;
        }
        x(0, 0) = 1;
        valid.X = x;
        validation_dataset.push_back(valid);
    }

    // train 
    for(int i = 0;i < num_epoch; i++) {
        // metrics_o << i + 1 << " ";
        for(int j = 0; j < train_dataset.size(); j++) {
            MatrixXf y = MatrixXf::Constant(10, 1, 0.0);
            y(train_dataset[j].label, 0) = 1.0;
            struct object O = NNForward(train_dataset[j].X, y, alpha, beta);
            struct grad gradient = NNBackward(O, train_dataset[j].X, y, beta);
            alpha = alpha - (gradient.grad_alpha) * learning_rate;
            beta = beta - (gradient.grad_beta) * learning_rate;
        }

        float train_entropy = 0;
        for(int j = 0; j < train_dataset.size(); j++) {
            MatrixXf y = MatrixXf::Constant(10, 1, 0.0);
            y(train_dataset[j].label, 0) = 1.0;
            MatrixXf y_hat = y_hat_compute(beta, alpha, train_dataset[j].X);
            for(int k = 0; k < 10; k++) {
                train_entropy += y(k, 0) * log(y_hat(k, 0));
            }
        }
        train_entropy /= train_dataset.size();
        train_entropy = -train_entropy;
        metrics_o << train_entropy << endl;

        // metrics2_o << i + 1 << " ";

        float valid_entropy = 0;
        for(int j = 0; j < validation_dataset.size(); j++) {
            MatrixXf y = MatrixXf::Constant(10, 1, 0.0);
            y(validation_dataset[j].label, 0) = 1.0;
            MatrixXf y_hat = y_hat_compute(beta, alpha, validation_dataset[j].X);
            for(int k = 0; k < 10; k++) {
                valid_entropy += y(k, 0) * log(y_hat(k, 0));
            }
        }
        valid_entropy /= validation_dataset.size();
        valid_entropy = -valid_entropy;
        metrics2_o << valid_entropy << endl;
    }


    // train_output
    for(int i = 0; i < train_dataset.size(); i++) {
        MatrixXf y = MatrixXf::Constant(10, 1, 0.0);
        struct object result = NNForward(train_dataset[i].X, y, alpha, beta);
        float maxx = -1;
        int label = -1;
        for(int j = 0; j < 10; j++) {
            if(result.y_hat(j, 0) > maxx) {
                maxx = result.y_hat(j, 0);
                label = j;
            }
        }
        if(label != train_dataset[i].label) {
            train_error += 1;
        }
        cout << label << endl;
    }
    train_o << alpha << endl;

    // validation_output
    for(int i = 0; i < validation_dataset.size(); i++) {
        MatrixXf y = MatrixXf::Constant(10, 1, 0.0);
        struct object result = NNForward(validation_dataset[i].X, y, alpha, beta);
        float maxx = -1;
        int label = -1;
        for(int j = 0; j < 10; j++) {
            if(result.y_hat(j, 0) > maxx) {
                maxx = result.y_hat(j, 0);
                label = j;
            }
        }
        if(label != validation_dataset[i].label) {
            valid_error += 1;
        }
        cout << label << endl;
    }
    valid_o << beta << endl;

    return 0;
}