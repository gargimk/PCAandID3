library("data.tree")
file.path <- readline(prompt="Enter path to dataset file: ")
dataset = read.csv(file=file.path,header=TRUE,sep=';')

IsPure <- function(data) {
 length(unique(data[,ncol(data)])) == 1
}

Entropy <- function( vls ) {
  res <- vls/sum(vls) * log2(vls/sum(vls))
  res[vls == 0] <- 0
  -sum(res)
}

InformationGain <- function( tble ) {
  tble <- as.data.frame.matrix(tble)
  entropyBefore <- Entropy(colSums(tble))
  s <- rowSums(tble)
  entropyAfter <- sum (s / sum(s) * apply(tble, MARGIN = 1, FUN = Entropy ))
  informationGain <- entropyBefore - entropyAfter
  return (informationGain)
}
buildID3Tree <- function(node, data) {
    
  node$obsCount <- nrow(data)
  
  #if the data-set is pure (e.g. all toxic), then
  if (IsPure(data)) {
    #construct a leaf having the name of the pure feature (e.g. 'toxic')
	child <- node$AddChild(unique(data[,ncol(data)]))
    node$feature <- tail(names(data), 1)
    child$obsCount <- nrow(data)
    child$feature <- ''
  } else {
    #chose the feature with the highest information gain (e.g. 'color')
    ig <- sapply(colnames(data)[-ncol(data)], 
            function(x) InformationGain(
              table(data[,x], data[,ncol(data)])
              )
            )
    feature <- names(ig)[ig == max(ig)][1]
    node$feature <- feature
	
    #take the subset of the data-set having that feature value
    childObs <- split(data[,!(names(data) %in% feature)], data[,feature], drop = TRUE)

    for(i in 1:length(childObs)) {
      #construct a child having the name of that feature value (e.g. 'red')

      child <- node$AddChild(names(childObs)[i])
      #call the algorithm recursively on the child and the subset      
      buildID3Tree(child, childObs[[i]])
    }
    
  }
  
  

}

Predict <- function(tree, features) {
if (tree$children[[1]]$isLeaf) return (tree$children[[1]]$name)
child <- tree$children[[features[[tree$feature]]]]
return ( Predict(child, features))
}

tree <- Node$new("dataset")
buildID3Tree(tree, dataset)
print(tree, "feature", "obsCount")
SetGraphStyle(tree, rankdir = "TB")
SetEdgeStyle(tree, arrowhead = "vee", color = "grey35", penwidth = 2)
SetNodeStyle(tree, style = "filled,rounded", shape = "box", fillcolor = "GreenYellow", 
            fontname = "helvetica", tooltip = GetDefaultTooltip)
Do(tree$leaves, function(node) SetNodeStyle(node, shape = "egg",fillcolor = "cyan"))
plot(tree)