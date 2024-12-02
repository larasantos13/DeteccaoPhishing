# Detecção de Phishing

1. O que é Phishing?

   Phishing é uma forma de fraude online em que criminosos tentam enganar vítimas para obter informações confidenciais, como senhas, números de cartões de crédito ou dados bancários. Esses ataques são geralmente realizados por meio de e-mails falsos, mensagens ou sites que se passam por fontes confiáveis.

2. Por que criei este projeto?

   Este projeto foi desenvolvido para ajudar empresas e usuários a identificar e evitar links e sites fraudulentos, oferecendo uma ferramenta confiável e acessível. Ele é direcionado a:
   
      Empresas: Para proteger funcionários e sistemas contra ameaças cibernéticas.
   
      Usuários individuais: Que necessitam de uma solução prática para analisar emails suspeitos.

4. Principais Problemas Identificados

      Crescimento de ataques sofisticados: Os ataques de phishing estão cada vez mais complexos, dificultando sua detecção por métodos tradicionais.
      Falta de ferramentas acessíveis: Há uma carência de soluções simples e eficazes para análise em tempo real de links suspeitos.

5. Como o Sistema Foi Desenvolvido

      4.1.Treinamento do Modelo:

   Utiliza-se redes neurais densas (Dense Neural Networks) para classificar textos de e-mails como "phishing" ou "seguro".
   Transformamos o texto dos e-mails em vetores usando TF-IDF, extraindo padrões relevantes para o treinamento.
   O modelo foi ajustado com técnicas como inicialização He e uma taxa de aprendizado otimizada, alcançando alta precisão nos resultados.

      4.2. Teste: 

   O sistema foi avaliado com um conjunto de e-mails inéditos, alcançando uma precisão de 99%.

6. Impacto do Projeto
   
   Redução de incidentes de phishing;
   Aumento da segurança organizacional;

7. Aprendizados e Habilidades Adquiridas
   
   Segurança cibernética: Aplicação de machine learning em problemas reais de segurança digital.
   Desenvolvimento de APIs: Criação de serviços robustos e eficientes usando Flask.
   Design de interfaces: Desenvolvimento de soluções acessíveis e intuitivas para usuários não técnicos.
   Aplicação prática de IA: Resolução de desafios reais por meio de inteligência artificial e aprendizado de máquina.

9. Melhoria contínua:

   Aperfeiçoamento da interface para facilitar a interpretação dos resultados, garantindo acessibilidade para usuários não técnicos.

Base de Dados utilizada para treinamento: https://www.kaggle.com/datasets/subhajournal/phishingemails
