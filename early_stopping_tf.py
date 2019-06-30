class Early_Stopping:
    
    def __init__(self, sess, saver, epochs_to_wait, metric_name):
        self.sess = sess
        self.saver = saver
        self.epochs_to_wait = epochs_to_wait
        self.metric_list = []
        self.counter = 0
        self.metric_name = metric_name
        
    def add_metric(self):
        self.metric_list.append(self.metric)
        
    def save_best_model(self,metric):
        self.metric = metric
        self.add_metric()
        
        if max(self.metric_list) == self.metric_list[-1]:
            self.best_metric = self.metric
            print('{} improved from {} to {}'.format(self.metric_name, self.metric_list[-2], self.best_metric))
            self.saver.save(self.sess, '/tmp/checkpoints/')
            self.counter = 0
            
        else:
            print('{} did not improve since test {} was {}'.format(self.metric_name, self.metric_name, self.best_metric))
            self.counter += 1

        
        
        
        