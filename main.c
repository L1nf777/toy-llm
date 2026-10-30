#include<stdio.h>
#include <math.h>

#define dim 3 //维度
#define V 2  //词表大小
#define max_input_length 8  //最大输入长度
#define d_k 3 //wQ,wK的大小
#define d_v 3  //wV的大小

//词表转化成编号
int token_to_id(char a)
{
    if(a == 'a')
    return 0;
    else if(a == 'b')
    return 1;
    else
    return -1;
}

float Embedding[V][dim] = {{0.5f,0.1f,-0.5f},{-0.5f,0.1f,0.5f}};

//查表函数search
int search(char text[], float x[][dim])
{
    int length = 0;
    while(text[length] != '\0')
    {
        if(length >= max_input_length)
        {
            printf("too long input.\n");
            return -1;
        }
        int id = token_to_id(text[length]);
        if(id == -1)
        {
            printf("invalid char.\n");
            return -1;
        }
        for(int d = 0; d < dim; d++)
        {
            x[length][d] = Embedding[id][d];
        }
        length++;
    }
    return length;
}

float pos_vec[max_input_length][dim] = {{0.0f,0.0f,0.0f},{0.1f,0.1f,0.1f},
{0.2f,0.2f,0.2f},{0.3f,0.3f,0.3f},{0.4f,0.4f,0.4f},{0.5f,0.5f,0.5f},{0.6f,0.6f,0.6f},{0.7f,0.7f,0.7f}};

//加上位置的影响
void add_pos(float x[][dim], int length)
{
    for(int t = 0; t < length; t++)
        for(int d = 0; d < dim; d++)
            x[t][d] += pos_vec[t][d];
}

//矩阵和向量的乘法
void linear(int d_in,int d_out,float vector[],float matrix[][d_out],float out[])
{
    for(int i = 0; i < d_out; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < d_in; j++)
        sum += vector[j] * matrix[j][i];
        out[i] = sum;
    }
}


float wQ[dim][d_k] = {{0.1f,0.2f,0.3f},{0.4f,0.5f,0.6f},{0.7f,0.8f,0.9f}};
float wK[dim][d_k] = {{-0.1f,-0.2f,-0.3f},{-0.4f,-0.5f,-0.6f},{-0.7f,-0.8f,-0.9f}};
float wV[dim][d_v] = {{0.1f,-0.2f,0.3f},{0.4f,-0.5f,0.6f},{-0.7f,0.8f,-0.9f}};

//向量点乘
float dot(float a[],float b[],int n)
{
    float sum = 0.0f;
    for(int i = 0; i < n; i++)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

//打分函数
float score(float a[],float b[],int n)
{
    float m = dot(a,b,n);
    float k = sqrt(n);
    return (m/k);
}

//归一的softmax
void softmax(float v[],int n)
{
    float sum = 0.0f;
    for(int i = 0; i < n; i++)
    {
        v[i] = expf(v[i]);
        sum += v[i];
    }
    for(int i = 0; i < n; i++)
    {
        v[i] = v[i]/sum;
    }
}

//交叉熵损失
float cross_entropy(float probs[],int correct_id)
{
    return -logf(probs[correct_id]);
}

#define N 7
//这里就是训练的语料
char* data[] = {"a","ab","aba","abab","ababa","ababab","abababa"};
char answer[] = {'b','a','b','a','b','a','b'};



float wOut[d_v][V] = {{0.1f,-0.3f},{0.4f,0.2f},{-0.5f,0.1f}};

//向前传播一次的函数
int forward(char text[], float probs[])
{
    float x[max_input_length][dim];
    int length = search(text,x);

    if(length <= 0)
        return -1;

    add_pos(x,length);
    float q[max_input_length][d_k];
    float k[max_input_length][d_k];
    float v[max_input_length][d_v];
    for(int i = 0;i < length; i++)
    {
        linear(dim,d_k,x[i],wQ,q[i]);
        linear(dim,d_k,x[i],wK,k[i]);
        linear(dim,d_v,x[i],wV,v[i]);
    }
    float s[length][length];
    for(int i = 0; i < length; i++)
    {
        for(int j = 0; j < length; j++)
        {
            if(j > i)
            s[i][j] = -1e9;
            else
            s[i][j] = score(q[i],k[j],d_k);
        }
    }

    for(int i = 0;i < length; i++)
    {
        softmax(s[i],length);
    }
    float out[length][d_v];
    for(int i = 0; i < length; i++)
    {
        linear(length,d_v,s[i],v,out[i]);
    }

    float logits[V];
    linear(d_v,V,out[length - 1],wOut,logits);
    softmax(logits,V);


    probs[0] = logits[0];
    probs[1] = logits[1];
    return length;
}
//根据给的训练数据（一条语料，切出的七个需要预测的题目）给出平均分
float avg_loss()
{
    float total = 0.0f;
    float probs[V];
    for(int n = 0; n < N; n++)
    {
        if(forward(data[n],probs) <= 0)
            return -1.0f;
        total += cross_entropy(probs,token_to_id(answer[n]));
    }
    return total/N;
}

//梯度函数，使用的差分近似，算完之后复原
float gradient(float* w)
{
    float eps = 0.01f;
    float old = *w;
    *w = old + eps;
    float L_plus = avg_loss();
    *w = old - eps;
    float L_minus = avg_loss();
    *w = old;
    return (L_plus - L_minus)/(2 * eps);
}


int main()
{
    float lr = 1.0f;//学习率，大小值得考虑。1.0:100步到0.0002；10:第3步NaN暴毙；0.1:过阻尼，100步才0.47
    float gEmbedding[V][dim];
    float gwQ[dim][d_k];
    float gwK[dim][d_k];
    float gwV[dim][d_v];
    float gwOut[d_v][V];

    //两遍for循环，第一次算出所有的梯度，第二次一次性沿梯度下降
    for(int step = 0; step < 100; step++)
    {
            for(int i = 0; i < V; i++)
        {
            for(int j = 0; j < dim; j++)
            {
                gEmbedding[i][j] = gradient(&Embedding[i][j]);
            }
        }
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < d_k; j++)
            {
                gwQ[i][j] = gradient(&wQ[i][j]);
            }
        }
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < d_k; j++)
            {
                gwK[i][j] = gradient(&wK[i][j]);
            }
        }
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < d_v; j++)
            {
                gwV[i][j] = gradient(&wV[i][j]);
            }
        }
        for(int i = 0; i < d_v; i++)
        {
            for(int j = 0; j < V; j++)
            {
                gwOut[i][j] = gradient(&wOut[i][j]);
            }
        }






        for(int i = 0; i < V; i++)
        {
            for(int j = 0; j < dim; j++)
            {
                Embedding[i][j] -= lr * gEmbedding[i][j];
            }
        }
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < d_k; j++)
            {
                wQ[i][j] -= lr * gwQ[i][j];
            }
        }
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < d_k; j++)
            {
                wK[i][j] -= gwK[i][j] * lr;
            }
        }
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < d_v; j++)
            {
                wV[i][j] -= gwV[i][j] * lr;
            }
        }
        for(int i = 0; i < d_v; i++)
        {
            for(int j = 0; j < V; j++)
            {
                wOut[i][j] -= gwOut[i][j] * lr;
            }
        }
        printf("step%d: loss = %.4f\n",step,avg_loss());
        
    }
    


    //接龙输出，挑概率大的接上去
    char text[1 + max_input_length] = "a";
    int length = 1;
    float probs[V];
    while(length < max_input_length)
    {
        forward(text, probs);
        char next = probs[0] > probs[1] ? 'a' : 'b';
        text[length] = next;
        length++;
        text[length] = '\0';
        printf("%s\n", text);
    }


    return 0;
}