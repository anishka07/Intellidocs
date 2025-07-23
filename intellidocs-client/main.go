package main

import (
	"context"
	"fmt"
	"log"
	"time"

	pb "github.com/anishka07/intellidocs-client/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)


type IntelliDocsClient struct {
	client pb.IntelliDocsServiceClient 
	conn *grpc.ClientConn
}


func NewIntelliDocsClient(serverAddress string) (*IntelliDocsClient, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, serverAddress,
	grpc.WithTransportCredentials(insecure.NewCredentials()),
	grpc.WithBlock(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to the server: %v", err)
	}
	
	client := pb.NewIntelliDocsServiceClient(conn)

	return &IntelliDocsClient{
		client: client,
		conn: conn,
	}, nil 
}


func (c *IntelliDocsClient) close() error {
	return c.conn.Close()
}


func (c *IntelliDocsClient) ProcessDocumnents(ctx context.Context, documentPaths []string) (*pb.ProcessDocumentResponse, error) {
	req := &pb.ProcessDocumentRequest{
		DocumentPaths: documentPaths,
	}

	log.Printf("Processing documents: %v", documentPaths)

	response, err := c.client.ProcessDocuments(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to process documents: %v", err)
	}

	return response, nil
}


func (c *IntelliDocsClient) RetrieveTopN(ctx context.Context, userQuery, docKey string, topN int32) (*pb.RetrieveTopNResponse, error) {
	req := &pb.RetrieveTopNRequest{
		UserQuery: userQuery,
		DocKey: docKey,
		TopN: topN,
	}

	log.Printf("retrieveing top %d results for query: '%s' in document: %s", topN, userQuery, docKey)

	response, err := c.client.RetrieveTopN(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve top N: %v", err)
	}

	return response, nil 
}


