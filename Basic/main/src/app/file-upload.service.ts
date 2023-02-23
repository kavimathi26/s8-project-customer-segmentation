import { Injectable } from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {Observable} from 'rxjs';
@Injectable({
providedIn: 'root'
})
export class FileUploadService {
	
// API url
//--baseApiUrl = "https://file.io"
	
constructor(private http:HttpClient) { }

server_add="http://127.0.0.1:5000"
// Returns an observable
//--upload(file):Observable<any> {

upload(file): Observable<any>{

	// Create form data
	const formData = new FormData();
		
	// Store form name as "file" with file data
	formData.append("file", file, file.name);
		
	// Make http post request over api
	// with formData as req
	//--return this.http.post(this.baseApiUrl, formData)
	return this.http.post(this.server_add,formData);
}
}
