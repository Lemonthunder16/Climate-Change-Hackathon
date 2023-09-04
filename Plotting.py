from matplotlib import pyplot as plt

#Plotting the loss vs number of epochs graph
def plot_the_loss_curve(epochs, mae_training, mae_validation):
    plt.figure()
    plt.title('Loss Curve')
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")

    #plot from 2nd epoch
    plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
    plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
    plt.legend()
    
    merged_mae_lists = mae_training[1:] + mae_validation[1:]
    highest_loss = max(merged_mae_lists)
    lowest_loss = min(merged_mae_lists)
    delta = highest_loss - lowest_loss
    print(delta) #decrease in loss

    top_of_y_axis = highest_loss + (delta * 0.05)
    bottom_of_y_axis = lowest_loss - (delta * 0.05)
    
    plt.ylim([bottom_of_y_axis, top_of_y_axis])
    plt.show()

def plot_the_data(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = df['Year']
    y = df['Month']
    z = df['Monthly_MSL']

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('Year')
    ax.set_ylabel('Month')
    ax.set_zlabel('Monthly MSL')

    plt.show()