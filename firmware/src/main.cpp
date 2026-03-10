#include <Arduino.h>
#include <ESP32Encoder.h>
#include "defines.h"
#include "rl_policy.h"

// Definições de pinos (ajuste conforme sua conexão)
#define ENC_RODA_A 17
#define ENC_RODA_B 16
#define ENC_PEND_A 19
#define ENC_PEND_B 18
#define IN_A 25
#define IN_B 26
#define ONBOARD_LED 2
#define BOTAO_SU 34
#define BOTAO_CONT 4

// Constantes do sistema
#define Ts 0.02f
#define en2rad_roda 0.006411413578755f
#define en2rad_pend 0.001570796326795f
#define pi 3.141592653589793f

// Comunicacao
#define DOUBLE_SIZE sizeof(double)
#define QTD_VARS_COM 5
#define TAMANHO_PAYLOAD (QTD_VARS_COM*sizeof(double))
#define TAMANHO_FRAME (TAMANHO_PAYLOAD + 2*sizeof(u_char))
void montaMensagem(u_char* msg, estados_t estados);

// Declaração dos handles das tasks
TaskHandle_t handleControle;
TaskHandle_t handleComunicacao;

// Criação de objeos p/ sincronia
SemaphoreHandle_t handleMutex = NULL;

// Ganho controlador
static const estados_t K{
  // .x1 = -60.2818,
  .x1 = -5.541321904960439,
  .x2 = -0.0,
  .x3 = -0.726359517978251,
  .x4 = -0.098019480640735
};
// static const estados_t K{
//   .x1 = -5.388444876977633f,
//   .x2 = 0.0,
//   .x3 = -0.796233738333759f,
//   .x4 = -0.086414484249118f
// };
bool enable_cont = false;
bool enable_rl = true;  // Toggle RL residual (set false for pure LQR)
void enable_cont_ISR(){
  enable_cont = !enable_cont;
}

// Definições para adquirir informações
// Condicoes iniciais
#define ACQSIGNAL false
#if ACQSIGNAL==true
#define acq_time 10.0 //s
#define acq_arr_size acq_time/Ts
const int arr_size = (int) acq_arr_size;
estados_t state_arr[arr_size];
bool acq_done = false;
int i = 0;
#endif

// Filtro e derivada
const double wf = 100; // rad/s
static const double backwards_uf(double q, double q_ant, double ts);
static const double backwards_filtered(double q, double q_ant, double qp_ant, double qp_ant_uf, double a, double b, double ts);
static const double sign(double x);

// Variaveis
// double angulo_0 = 0.0f, theta_1 = 0.0f, angulo = 0.0f;
estados_t estados{
  .x1 = 0.0f,
  .x2 = 0.0f,
  .x3 = 0.0f,
  .x4 = 0.0f
};
estados_t estados_ant{
  .x1 = 0.0f,
  .x2 = 0.0f,
  .x3 = 0.0f,
  .x4 = 0.0f
};
estados_t estados_ant_unf{
  .x1 = 0.0f,
  .x2 = 0.0f,
  .x3 = 0.0f,
  .x4 = 0.0f
};

double u_hist=0;

double ajusta_angulo(double angulo);
double controle(estados_t x, estados_t k);

 
// Objetos globais
ESP32Encoder enc_roda;
ESP32Encoder enc_pend;
hw_timer_t *timer = NULL;
volatile bool aux = false;


// Configuração PWM
const int pwmChannelA = 0;
const int pwmChannelB = 1;
const int pwmResolution = 8;
const int pwmFrequency = 10000;

void IRAM_ATTR onTimer() {
  aux = true;
}

void drive(float u) {
  int direction = 0;
  if (u > 0) direction = 1;
  if (u < 0) direction = 2;

  u = fabs(u);
  u = constrain(u, 0.0f, 1.0f);
  //ledcWrite(pwmChannel, (uint32_t)(u * 255));

  switch(direction) {
    case 0:
      ledcWrite(pwmChannelA, (uint32_t)(0 * 255));
      ledcWrite(pwmChannelB, (uint32_t)(0 * 255));
      break;
    case 1:
    
      ledcWrite(pwmChannelA, (uint32_t)(u * 255));
      ledcWrite(pwmChannelB, (uint32_t)(0 * 255));
      break;
    case 2:
      ledcWrite(pwmChannelA, (uint32_t)(0 * 255));
      ledcWrite(pwmChannelB, (uint32_t)(u * 255));
      break;
  }
}

void taskControle(void* pvParameters);
void taskComunicacao(void* pvParameters);

void setup() {
  Serial.begin(9600);

  // ConfiguraÇão botoes
  pinMode(BOTAO_CONT, INPUT_PULLDOWN);
  pinMode(BOTAO_SU, INPUT_PULLDOWN);
  attachInterrupt(digitalPinToInterrupt(BOTAO_CONT), enable_cont_ISR, RISING);

  // Configuração dos encoders
  pinMode(ENC_RODA_A, INPUT);
  pinMode(ENC_RODA_B, INPUT);
  ESP32Encoder::useInternalWeakPullResistors = puType::up;
  enc_roda.attachFullQuad(ENC_RODA_A, ENC_RODA_B);
  enc_pend.attachFullQuad(ENC_PEND_A, ENC_PEND_B);
  enc_roda.clearCount();
  enc_pend.clearCount();

  // Configuração do motor
  ledcSetup(pwmChannelA, pwmFrequency, pwmResolution);
  ledcSetup(pwmChannelB, pwmFrequency, pwmResolution);
  ledcAttachPin(IN_A, pwmChannelA);
  ledcAttachPin(IN_B, pwmChannelB);

  // LED onboard
  pinMode(ONBOARD_LED, OUTPUT);
  digitalWrite(ONBOARD_LED, HIGH);

  // Configuração do timer
  timer = timerBegin(0, 80, true); // Divisor de 80 (1MHz)
  timerAttachInterrupt(timer, &onTimer, true);
  timerAlarmWrite(timer, Ts * 1000000, true);
  timerAlarmEnable(timer);
  
  // Pega ângulo inicial
  Serial.println("");
  Serial.println("-----------------------");
  Serial.println("-------Iniciando!------");
  Serial.println("-----------------------");

  // Configura mutex
  handleMutex = xSemaphoreCreateMutex();

  delay(1000); // Espera inicial de 5 segundos

  // Configura taks
  xTaskCreatePinnedToCore(
    taskControle,
    "TaskControle",
    4096,  // Increased for RL inference stack (~300B for hidden layers)
    NULL,
    1,
    &handleControle,
    1
  );

  xTaskCreatePinnedToCore(
    taskComunicacao,
    "TaskComunicacao",
    1000,
    NULL,
    1,
    &handleComunicacao,
    0
  );
}

void taskControle(void* pvParameteres){
  for(;;){
    if(aux) {
      aux = false;
      if(xSemaphoreTake(handleMutex, portMAX_DELAY)) {


        estados.x1 = enc_pend.getCount() * en2rad_pend;
        estados.x2 = enc_roda.getCount() * en2rad_roda;
        estados.x3 = backwards_uf(estados.x1, estados_ant.x1, Ts);
        estados.x4 = backwards_uf(estados.x2, estados_ant.x2, Ts);

        estados_ant = estados;

        estados.x1 = ajusta_angulo(enc_pend.getCount() * en2rad_pend);

        double u = controle(estados, K);

        // Residual RL: cogging torque compensation
        if(enable_rl) {
          u += rl_inference(estados.x1, estados.x2, estados.x3, estados.x4);
        }

        if(abs(u) > 12) u = sign(u)*12;
        if(!enable_cont) u=0.0f;
        u_hist = u;
        drive(u);
        xSemaphoreGive(handleMutex);
      }
    }
  }
}

void taskComunicacao(void* pvParameters){
  estados_t X_com = {
    .x1 = 0.0,
    .x2 = 0.0,
    .x3 = 0.0,
    .x4 = 0.0,
  };
  u_char* msg = (u_char*) malloc(TAMANHO_FRAME);
  for(;;){
    
    if(xSemaphoreTake(handleMutex, 200 * portTICK_PERIOD_MS)){
      X_com = estados;
      xSemaphoreGive(handleMutex);
      montaMensagem(msg, X_com);
      Serial.write(msg, TAMANHO_FRAME);
    }
    delay(100);
  }
  free(msg);
};

void loop() {}

static const double backwards_uf(double q, double q_ant, double ts){
  return 1.0f/Ts*(q - q_ant);
}
static const double backwards_filtered(double q, double q_ant, double qp_ant, double qp_ant_uf, double a, double b, double ts){
  return a*(backwards_uf(q, q_ant, ts) + qp_ant_uf) - b*qp_ant;
}

static const double sign(double x){
  return x!=0 ? x/abs(x) : 0;
}

double ajusta_angulo(double angulo){
  if(abs(angulo) > 2*pi){
    angulo = fmod(angulo, 2*pi);
  }
  double angulo_pra_cima = angulo - sign(angulo)*pi;
  return angulo_pra_cima;
}

double controle(estados_t x, estados_t k){
  return -k.x1 * x.x1 -k.x2 * x.x2 -k.x3 * x.x3 -k.x4 * x.x4;
}

void montaMensagem(u_char* msg, estados_t estados){
  msg[0] = 0xFF;
  memcpy(msg + 1 + 0*DOUBLE_SIZE, &estados.x1, DOUBLE_SIZE);
  memcpy(msg + 1 + 1*DOUBLE_SIZE, &estados.x2, DOUBLE_SIZE);
  memcpy(msg + 1 + 2*DOUBLE_SIZE, &estados.x3, DOUBLE_SIZE);
  memcpy(msg + 1 + 3*DOUBLE_SIZE, &estados.x4, DOUBLE_SIZE);
  memcpy(msg + 1 + 4*DOUBLE_SIZE, &u_hist, DOUBLE_SIZE);
  msg[1 + TAMANHO_PAYLOAD] = 0x00;
}